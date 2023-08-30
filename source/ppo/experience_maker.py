import json
import time
from collections import defaultdict
from copy import deepcopy
from functools import partial
from json import JSONDecodeError
from typing import List

import ray
import torch
from colorama import Fore, Style
from openrlhf.models.utils import masked_mean, unpacking_samples
from openrlhf.trainer import ppo_utils
from tqdm import tqdm
from vllm import RequestOutput

from source.feedback.annotate import (assess_parse, build_assess_prompt, build_extract_prompt,
                                      build_verify_prompt, extract_parse, verify_parse)
from source.feedback.reward import Reward, RewardConfig, post_process
from source.ppo.ppo_utils import Experience, Samples, compute_approx_kl, compute_reward


class RemoteExperienceMaker(ppo_utils.RemoteExperienceMaker):
    def __init__(
        self,
        *args,
        vllm_engines: List = None,
        packing_samples=False,
        reward_config: RewardConfig = RewardConfig(),
        **kwargs
    ):
        super().__init__(*args, vllm_engines=vllm_engines, packing_samples=packing_samples, **kwargs)
        self.config = reward_config
        self.reward = Reward(self.config)
        if self.config.annotator_config.completion_config.model_url is not None:
            self._get_rewards = self._get_rewards_remote
        else:
            self._get_rewards = self._get_rewards_local

    @torch.no_grad()
    def make_experience_list(
        self,
        all_prompts: List[str],
        titles: List[List[str]],
        materials: List[List[str]],
        extra_materials: List[List[str]],
        **generate_kwargs
    ) -> List[Experience]:
        """
        Make a list of experience with the micro_rollout_batch_size.

        This method will first calculate the response sequences and rewards for the given prompts.
        Then, if we need certain processing for the rewards or do certain filtering, we can process the rollout as a whole.
        After that, we will calculate the advantages and returns for each experience.
        """
        if self.strategy.args.perf:
            self.perf_stats = {
                "generate_time": 0,
                "actor_value_rm_time": 0,
                "wait_time": 0,
            }
        # NOTE: Here all_prompts do not have chatml format
        all_outputs = self._get_responses(all_prompts, **generate_kwargs)
        all_rewards = self._get_rewards(
            all_prompts=all_prompts,
            all_outputs=all_outputs,
            titles=titles,
            materials=materials,
            extra_materials=extra_materials,
        )
        # Make sure all requests are sent.
        torch.distributed.barrier()
        # vLLM offload when colocate_all_models
        if self.strategy.args.vllm_enable_sleep:
            if torch.distributed.get_rank() == 0:
                refs = []
                for engine in self.vllm_engines:
                    refs.append(engine.sleep.remote())
                ray.get(refs)

        samples_list = self.pack_into_samples(all_outputs, all_rewards)
        experiences = []
        for samples in tqdm(
            samples_list, desc="make_experience",
            disable=not self.strategy.is_rank_0(),
        ):
            experiences.append(self.make_experience(samples))
        experiences = self.process_experiences(experiences)[0]

        args = self.strategy.args
        # calculate return and advantages
        for experience in experiences:
            experience.to_device("cuda")
            reward = experience.reward
            num_actions = experience.info["num_actions"]
            reward = compute_reward(  # MONKEY: Rewrite here
                reward,
                self.kl_ctl.value,
                experience.kl,
                action_mask=experience.action_mask,
                num_actions=num_actions,
                reward_clip_range=args.reward_clip_range,
            )

            if self.advantage_estimator == "gae":
                experience.advantages, experience.returns = self.get_advantages_and_returns(
                    experience.values,
                    reward,
                    experience.action_mask,
                    generate_kwargs["gamma"],
                    generate_kwargs["lambd"],
                )
            elif self.advantage_estimator == "reinforce":
                experience.returns = self.get_cumulative_returns(
                    reward,
                    experience.action_mask,
                    generate_kwargs["gamma"],
                )
                experience.advantages = deepcopy(experience.returns)
            else:
                raise Exception(f"Unkown advantage_estimator {self.advantage_estimator}")

            # calculate the return info.
            if not getattr(self, "packing_samples", False):
                return_sums = reward.sum(dim=-1)
            else:
                return_sums = torch.tensor(
                    [each_reward.sum() for each_reward in reward], device=torch.cuda.current_device()
                )
            experience.info["return"] = return_sums
            # remove unnecessary info
            experience.kl = None
            del experience.info["num_actions"]

        if self.critic is not None:
            for experience in experiences:
                # send experience to critic
                experience_cpu = deepcopy(experience)
                experience_cpu.to_device("cpu")
                self._ref = self.critic.append.remote(experience_cpu)
        return experiences

    @torch.no_grad()
    def make_experience(self, samples: Samples) -> Experience:
        """
        Turn samples into experience by calculating logprobs, values, rewards, and kl divergence.
        """
        # rank = torch.distributed.get_rank()
        args = self.strategy.args
        self.actor.eval()
        device = torch.cuda.current_device()

        # extract values from samples
        sequences = samples.sequences
        attention_mask = samples.attention_mask
        action_mask = samples.action_mask
        num_actions = samples.num_actions
        packed_seq_lens = samples.packed_seq_lens

        start = time.time()
        sequences_cpu, attention_mask_cpu = (
            sequences.to("cpu"),
            attention_mask.to("cpu"),
        )

        # init log probs
        base_action_log_probs_ref = self.initial_model.forward.remote(
            sequences_cpu, num_actions, attention_mask_cpu, packed_seq_lens=packed_seq_lens
        )

        if args.colocate_all_models:
            # print(f"[INFO {rank=}] Get base_action_log_probs_ref ")
            ray.get([base_action_log_probs_ref])
            ray.get([self.initial_model.empty_cache.remote()])
            # print(f"[INFO {rank=}] Get base_action_log_probs_ref DONE")

        # values
        if self.critic is not None:
            value_ref = self.critic.forward.remote(
                sequences_cpu, num_actions, attention_mask_cpu, packed_seq_lens=packed_seq_lens
            )
            # avoid CUDA OOM when colocate models
            if args.colocate_critic_reward or args.colocate_all_models:
                # print(f"[INFO {rank=} Get value_ref")
                ray.get([value_ref])
                ray.get([self.critic.empty_cache.remote()])
                # print(f"[INFO {rank=} Get value_ref DONE")
        else:
            value_ref = ray.put(None)

        if args.colocate_actor_ref:
            # print(f"[INFO {rank=} Get base_action_log_probs_ref")
            ray.get([base_action_log_probs_ref])
            ray.get([self.initial_model.empty_cache.remote()])
            # print(f"[INFO {rank=} Get base_action_log_probs_ref DONE")

        # log probs
        # print(f"[INFO {rank=} Get action_log_probs")
        action_log_probs = self.actor(sequences, num_actions, attention_mask, packed_seq_lens=packed_seq_lens)
        # print(f"[INFO {rank=} Get action_log_probs DONE")
        actor_value_rm_time = time.time() - start

        # wait initial/critic/reward model done
        start = time.time()
        # print(f"[INFO {rank=} Get ref_values")
        ref_values = ray.get([base_action_log_probs_ref, value_ref])
        # print(f"[INFO {rank=} Get ref_values DONE")
        wait_time = time.time() - start

        # NOTE: We do not use a seperate reward model anymore
        base_action_log_probs, value = ref_values[0], ref_values[1]
        base_action_log_probs = base_action_log_probs.to(device)
        if value is not None:
            value = value.to(device)

        reward = samples.rewards

        if args.colocate_actor_ref or args.colocate_all_models:
            torch.cuda.empty_cache()

        kl = compute_approx_kl(
            action_log_probs,
            base_action_log_probs,
            action_mask=action_mask,
            use_kl_estimator_k3=args.use_kl_estimator_k3,
        )

        if not self.packing_samples:
            kl_mean = masked_mean(kl, action_mask, dim=-1)
        else:
            # convert tensor into list of tensors so that it's easier to manipulate
            # within dataset.
            sequences = unpacking_samples(sequences, packed_seq_lens)
            attention_mask = None
            action_log_probs = unpacking_samples(action_log_probs, num_actions)
            if value is not None:
                value = unpacking_samples(value, num_actions)

            reward = unpacking_samples(reward, num_actions)
            kl = unpacking_samples(kl, num_actions)
            kl_mean = torch.tensor([each_kl.mean() for each_kl in kl], device=device)

        r = torch.tensor([each_reward.sum() for each_reward in reward], device=device)

        info = {
            "kl": kl_mean,
            "reward": r,
            "response_length": samples.response_length,
            "total_length": samples.total_length,
            "num_actions": num_actions,
        }

        if self.strategy.args.perf:
            self.perf_stats["actor_value_rm_time"] += actor_value_rm_time
            self.perf_stats["wait_time"] += wait_time

        experience = Experience(
            sequences=sequences,
            action_log_probs=action_log_probs,
            values=value,
            returns=None,
            advantages=None,
            attention_mask=attention_mask,
            action_mask=action_mask,
            info=info,
            kl=kl,
            reward=reward,
        )

        self.actor.train()  # reset model state
        return experience

    def _get_responses(self, all_prompts: List[str], **kwargs):
        all_prompts = sum([[prompt] * self.strategy.args.n_samples_per_prompt for prompt in all_prompts], [])
        return self._generate_vllm(all_prompts, **kwargs)

    def _get_rewards_remote(
        self,
        all_prompts: List[str],
        all_outputs: List[RequestOutput],
        titles: List[List[str]],
        materials: List[List[str]],
        extra_materials: List[List[str]],
        **kwargs,
    ):
        questions = [ques.strip() for ques in all_prompts]
        answers = [output.outputs[0].text for output in all_outputs]
        sample_outputs = [output.outputs[0].token_ids for output in all_outputs]
        all_labels = self.reward.annotator.annotate(
            answers=answers,
            questions=questions,
            titles=titles,
            materials=materials,
            extra_materials=extra_materials
        )
        all_rewards = self.reward.labels_to_rewards(
            labels=all_labels,
            sample_outputs=sample_outputs,
            tokenizer=self.tokenizer,
            granularity=self.config.granularity,
        )
        # NOTE: all_rewards是一个和list of tensors和输出tensors一样
        # 采用这个形状是为了和compute_reward函数兼容
        # 而在pack_into_samples我们完成相关的padding逻辑
        return all_rewards

    def _get_rewards_local(
        self,
        all_prompts: List[str],
        all_outputs: List[RequestOutput],
        titles: List[List[str]],
        materials: List[List[str]],
        extra_materials: List[List[str]],
        max_retry_times: int = 4,
        **kwargs,
    ):
        questions = [ques.strip() for ques in all_prompts]
        answers = [output.outputs[0].text for output in all_outputs]
        sample_outputs = [output.outputs[0].token_ids for output in all_outputs]
        annotations = [
            defaultdict(lambda: defaultdict(dict))
            for _ in range(len(answers))
        ]
        materials = materials or [[] for _ in range(len(answers))]
        extra_materials = extra_materials or [None for _ in range(len(answers))]
        n_samples_per_prompt = self.strategy.args.n_samples_per_prompt

        tasks = [{
            "task_id": i, "task_name": "extract", "retry_times": 0,
            "build_task_prompt": partial(build_extract_prompt, answer=ans),
        } for i, ans in enumerate(answers)]
        while len(tasks) > 0:
            task_prompts = [task["build_task_prompt"]() for task in tasks]
            generations = self._generate_vllm(task_prompts, **kwargs)

            new_tasks = []
            for i, generation in enumerate(generations):
                generation = generation.outputs[0].text
                task = tasks[i]
                task_id, task_name = task['task_id'], task['task_name']
                try:
                    if task_name == "extract":
                        result = extract_parse(generation)
                        for sentence, statements in result.items():
                            for statement in statements:
                                new_tasks.append({
                                    "task_id": task_id,
                                    "task_name": 'verify',
                                    "sentence": sentence,
                                    "statement": statement,
                                    "build_task_prompt": partial(
                                        build_verify_prompt,
                                        question=questions[task_id // n_samples_per_prompt],
                                        statement=statement,
                                        titles=titles[task_id // n_samples_per_prompt],
                                        materials=materials[task_id // n_samples_per_prompt],
                                        retrieval=self.reward.annotator.retrieval,
                                        extra_materials=extra_materials[task_id // n_samples_per_prompt],
                                    ),
                                    "retry_times": 0,
                                })
                                new_tasks.append({
                                    "task_id": task_id,
                                    "task_name": 'assess',
                                    "sentence": sentence,
                                    "statement": statement,
                                    "build_task_prompt": partial(
                                        build_assess_prompt,
                                        question=questions[task_id // n_samples_per_prompt],
                                        statement=statement,
                                        answer=answers[task_id],
                                    ),
                                    "retry_times": 0,
                                })
                    else:
                        parse = assess_parse if task_name == "assess" else verify_parse
                        annotations[task_id][task['sentence']][task['statement']][task_name] = parse(generation)

                except JSONDecodeError:
                    task['retry_times'] += 1
                    if task['retry_times'] >= max_retry_times:
                        if task_name == "verify":
                            annotations[task_id][task['sentence']][task['statement']][task_name] = 'Vague'
                        elif task_name == "assess":
                            annotations[task_id][task['sentence']][task['statement']][task_name] = '3'
                    else:
                        new_tasks.append(task)

            tasks = new_tasks

        all_labels = [annotations[k] for k in range(len(answers))]
        for i, annotation in enumerate(all_labels):
            for sentence, statements in annotation.items():
                for statement, result in statements.items():
                    all_labels[i][sentence][statement] = [result['verify'], result['assess']]

        self.strategy.print(
            f"{Fore.GREEN}QUESTION:{Style.RESET_ALL}\n{questions[0]}\n"
            f"{Fore.GREEN}ANSWER:{Style.RESET_ALL}\n{answers[0]}\n"
            f"{Fore.GREEN}ANNOTATED:{Style.RESET_ALL}\n{json.dumps(all_labels[0], indent=2, ensure_ascii=False)}\n"
            f"{Fore.GREEN}MATERIALS:{Style.RESET_ALL} {sum(map(len, materials))} words"
        )

        all_labels = post_process(all_labels)
        all_rewards = self.reward.labels_to_rewards(
            labels=all_labels,
            sample_outputs=sample_outputs,
            tokenizer=self.tokenizer,
            granularity=self.config.granularity,
        )
        # NOTE: all_rewards是一个和list of tensors和输出tensors一样
        # 采用这个形状是为了和compute_reward函数兼容
        # 而在pack_into_samples我们完成相关的padding逻辑
        return all_rewards

    def _generate_vllm(self, all_prompts: List[str], **kwargs) -> List[str]:
        from vllm import SamplingParams

        # round-robin load balance
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()

        # Select LLM engines: assign each rank an engine, or cycle through engines if world_size < engine_count
        if len(self.vllm_engines) <= world_size:
            llms = [self.vllm_engines[rank % len(self.vllm_engines)]]
        else:
            llms = self.vllm_engines[rank::world_size]

        sampling_params = SamplingParams(
            temperature=kwargs.get("temperature", 1.0),
            top_p=kwargs.get("top_p", 1.0),
            top_k=kwargs.get("top_k", -1),
            max_tokens=kwargs.get("max_new_tokens", 1024),
            min_tokens=kwargs.get("min_new_tokens", 1),
            skip_special_tokens=kwargs.get("skip_special_tokens", False),
        )

        # Expand prompt list based on the number of samples per prompt
        # all_prompt_token_ids = self.tokenize_fn(all_prompts, self.prompt_max_len, padding=False)["input_ids"]
        all_prompts = [self.tokenizer.apply_chat_template(  # NOTE: Where we apply chat template
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            max_length=self.prompt_max_len,
            padding=False,
        ) for prompt in all_prompts]

        # Distribute requests to engines and collect responses to outputs
        all_output_refs = []
        batch_size = (len(all_prompts) + len(llms) - 1) // len(llms)
        for i, llm in enumerate(llms):
            prompts = all_prompts[i * batch_size: (i + 1) * batch_size]
            if prompts:
                all_output_refs.append(
                    llm.generate.remote(sampling_params=sampling_params, prompts=prompts)
                )

        # Retrieve and combine results from all outputs
        all_outputs = sum(ray.get(all_output_refs), [])
        return all_outputs

    def pack_into_samples(self, all_outputs, all_rewards):
        samples_list = []
        for i in range(0, len(all_outputs), self.strategy.args.micro_rollout_batch_size):
            outputs = all_outputs[i: i + self.strategy.args.micro_rollout_batch_size]
            rewards = all_rewards[i: i + self.strategy.args.micro_rollout_batch_size]
            if not self.packing_samples:
                # NOTE: concat all outputs to following format:
                #
                # | [PAD] [PAD] token token token | token token [EOS] [PAD] |
                # | token token token token token | token token [EOS] [PAD] |
                # | [PAD] [PAD] [PAD] token token | token token token [EOS] |
                # |<---------- prompt ----------->|<-------- answer ------->|
                max_input_len, max_output_len = 0, 0
                for output in outputs:
                    max_input_len = max(max_input_len, len(output.prompt_token_ids))
                    max_output_len = max(max_output_len, len(output.outputs[0].token_ids))

                pad_token_id, eos_token_id = self.tokenizer.pad_token_id, self.tokenizer.eos_token_id
                sequences, padded_rewards = [], []
                for i, output in enumerate(outputs):
                    # left padding input
                    input_len = len(output.prompt_token_ids)
                    input_ids = [pad_token_id] * (max_input_len - input_len) + list(output.prompt_token_ids)

                    # right padding output
                    output_len = len(output.outputs[0].token_ids)
                    output_ids = list(output.outputs[0].token_ids) + [pad_token_id] * (max_output_len - output_len)
                    reward_len = len(rewards[i])
                    reward = rewards[i] + [0] * (max_output_len - output_len)  # NOTE: Only padding to output_len

                    if output_ids[output_len - 1] != eos_token_id:
                        output_ids[min(output_len, len(output_ids) - 1)] = eos_token_id
                        reward[min(reward_len, len(output_ids) - 1)] -= 1  # NOTE: Penalize for not ending with EOS

                    # concat input and output
                    sequences.append(input_ids + output_ids)
                    padded_rewards.append(reward)

                sequences = torch.tensor(sequences)
                sequences, attention_mask, action_mask = self.actor.process_sequences(
                    sequences, max_input_len, eos_token_id, pad_token_id
                )
                sequences = sequences.to("cuda")
                padded_rewards = torch.as_tensor(padded_rewards, device="cuda")
                attention_mask = attention_mask.to("cuda")
                action_mask = action_mask.to("cuda")
                samples_list.append(
                    Samples(
                        sequences=sequences,
                        rewards=padded_rewards,  # NOTE: We expect all_rewards to be same size with sequences
                        attention_mask=attention_mask,
                        action_mask=action_mask,
                        num_actions=action_mask.size(1),
                        packed_seq_lens=None,
                        response_length=action_mask.float().sum(dim=-1),
                        total_length=attention_mask.float().sum(dim=-1),
                    )
                )
            else:
                # NOTE: concat all outputs to following format:
                #
                # | token token token | token token [EOS] | token token token token token | token token [EOS] | token token | token token token [EOS] |
                # |<---  prompt ----->|<---- answer ----->|<---------- prompt ----------->|<----- answer ---->|<- prompt -->|<-------- answer ------->|
                pad_token_id, eos_token_id = self.tokenizer.pad_token_id, self.tokenizer.eos_token_id
                sequences = []
                padded_rewards = []
                packed_seq_lens = []
                attention_mask = []
                num_actions = []
                for i, output in enumerate(outputs):
                    input_len = len(output.prompt_token_ids)
                    output_len = len(output.outputs[0].token_ids)
                    packed_seq_lens.append(input_len + output_len)
                    sequences.extend(output.prompt_token_ids + list(output.outputs[0].token_ids))
                    padded_rewards.extend(rewards[i])  # NOTE: Keep output size
                    attention_mask.extend([i + 1] * (input_len + output_len))

                    # current_action_mask = [0] * (input_len - 1) + [1] * output_len + [0]
                    # num_actions.append(max(1, sum(current_action_mask)))
                    num_actions.append(max(1, output_len))

                sequences = torch.tensor(sequences, device="cuda").unsqueeze(0)
                padded_rewards = torch.tensor(padded_rewards, device="cuda").unsqueeze(0)
                attention_mask = torch.tensor(attention_mask, device="cuda").unsqueeze(0)
                action_mask = None
                response_length = torch.tensor(num_actions, device="cuda", dtype=torch.float)
                total_length = torch.tensor(packed_seq_lens, device="cuda", dtype=torch.float)
                samples_list.append(
                    Samples(
                        sequences=sequences,
                        rewards=padded_rewards,
                        attention_mask=attention_mask,
                        action_mask=None,
                        num_actions=num_actions,
                        packed_seq_lens=packed_seq_lens,
                        response_length=response_length,
                        total_length=total_length,
                    )
                )
        return samples_list
