import argparse

import gradio as gr
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from source.dataset import format_qa_pair

EXAMPLE = """
Check the factuality and helpfulness of a response to a question based on the materials.
- Extraction: Break the sentences with object facts into atomic statements.
- Truthfulness Verification: verify each statement based on the materials.
    - "Correct": The statement is proved by the materials.
    - "Hedged correct": The statement is expressed with uncertainty but is true.
    - "Vague": The statement can not be decided as true or false based on the materials.
    - "Hedged wrong": The statement is expressed with uncertainty and is false.
    - "Wrong": The statement is denied by the materials.
- Informativeness Assessment: evaluate the helpfulness of each statement.
    - "5": The statement is an answer to the question.
    - "4": The statement provides crucial information.
    - "3": The statement gives relevant content.
    - "2": The statement contains supplementary facts.
    - "1": The statement is repeating information, providing useless information.
- Details:
    - Do not consider the correctness of the statement when assessing informativeness.
    - Follow the valid YAML format.
    - Skip the sentences like "It's hard to say ...", "I'm sorry, I don't know ...". If there is no valid statement, output {}

>> Example:
[Materials]
- First for Women is a women's magazine published by Bauer Media Group in the USA. The magazine was started in 1989. It is based in Englewood Cliffs, New Jersey. In 2011 the circulation of the magazine was 1,310,696 copies."
- Arthur's Magazine (1844–1846) was an American literary periodical published in Philadelphia in the 19th century. Edited by T.S. Arthur, it featured work by Edgar A. Poe, J.H. Ingraham, Sarah Josepha Hale, Thomas G. Spear, and others. In May 1846 it was merged into "Godey's Lady's Book".
- Which magazine was started first Arthur's Magazine or First for Women? The correct answer may be "Arthur's Magazine".
[Question]
Which magazine was started first, Arthur's Magazine founded by Arthur K. Watson, or First for Women?
[Response]
It is difficult to say which game has been released in more versions without more information, so I can only guess based on my training data. Arthur's Magazine was likely started first. It was possibly founded in 1923 by Arthur K. Watson, a prominent publisher in the field of men's magazines. First for Women, on the other hand, was not founded until 1989. It was created as a spin-off of Family Circle magazine, which was founded in 1957.
[Annotation]:
```yaml
Arthur's Magazine was likely started first.:
  Arthur's Magazine was likely started first.:
    - Correct
    - 5
'It was possibly founded in 1923 by Arthur K. Watson, a prominent publisher in the field of men''s magazines.':
  It was possibly founded in 1923.:
    - Wrong
    - 4
  It was founded by Arthur K. Watson.:
    - Wrong
    - 3
  Arthur K. Watson is a prominent publisher in the field of men's magazines.:
    - Vague
    - 2
'First for Women, on the other hand, was not founded until 1989.':
  First for Women was not founded until 1989.:
    - Correct
    - 4
'It was created as a spin-off of Family Circle magazine, which was founded in 1957.':
  It was created as a spin-off of Family Circle magazine.:
    - Vague
    - 3
  Family Circle magazine was founded in 1957.:
    - Vague
    - 2
```

>> Real Problem:
[Materials]
- Boogie (Spanish: \"Boogie, el aceitoso\" ) is a 2009 3D Argentinian Flash-animated action-thriller film, based on the Argentine character Boogie, the oily by Roberto Fontanarrosa, and directed by Gustavo Cova. The voices of main characters Boogie and Marcia were performed by Pablo Echarri and Nancy Dupláa. It was the first 3D animated movie made in Argentina and Latin America.
- Monsters vs. Aliens is a 2009 American 3D computer-animated science fiction film produced by DreamWorks Animation and distributed by Paramount Pictures. It was DreamWorks Animation's first feature film to be directly produced in a stereoscopic 3-D format instead of being converted into 3-D after completion, which added $15 million to the film's budget.
- The correct answer for \"Are the movies Boogie and Monsters vs. Aliens both 3D animated films\" is \"yes\".
[Question]
Are the movies Boogie and Monsters vs. Aliens both 3D animated films?
[Response]
Yes, both \"Boogie\" and \"Monsters vs. Aliens\" are 3D animated films. \"Boogie\" is a 2021 American computer-animated sports comedy film directed by Chris Bailey and produced by Walt Disney Pictures. It tells the story of a young boy named Boogie, who dreams of becoming a professional basketball player, and his journey to the NBA.\n\n\"Monsters vs. Aliens\" is a 2009 American 3D computer-animated science fiction comedy film produced by DreamWorks Animation and distributed by Paramount Pictures. The film follows the story of a woman named Susan who is abducted by aliens and taken to their planet, where she meets a group of monsters who team up to defend Earth from an alien invasion.
[Annotation]
"""


def http_bot(prompt):
    prompt = format_qa_pair(prompt, tokenizer=tokenizer)
    outputs = llm.generate([prompt], sampling_params)

    for output in outputs:
        generated_text = output.outputs[0].text
        return generated_text.replace(prompt, "").strip()


def build_demo():
    with gr.Blocks() as demo:
        gr.Markdown("# vLLM text completion demo\n")
        input_box = gr.Textbox(label="Input",
                               placeholder="Enter text and press ENTER")
        output_box = gr.Textbox(label="Output",
                                placeholder="Generated result from the model")
        input_box.submit(http_bot, [input_box], [output_box])
    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-70B-Instruct")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(model=args.model)
    sampling_params = SamplingParams(max_tokens=4096, temperature=0.1, top_p=0.1)

    gr_demo = build_demo()
    gr_demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=False
    )
