import jsonlines
from tqdm import tqdm

from source.feedback.annotate import AnnotatorConfig, CompletionConfig
from source.utils import count_lines


def main(
    data_path: str = 'experiments/outputs/trained.jsonl',
    output_path: str = 'experiments/outputs/trained-annotated.jsonl',
    annotator_config: AnnotatorConfig = AnnotatorConfig(
        num_procs=512,
        completion_config=CompletionConfig(
            model_url="http://127.0.0.1:8000/v1",
            model="Qwen/Qwen2.5-72B-Instruct",
        )
    ),
):
    with jsonlines.open(data_path, 'r') as reader:
        dataset = list(tqdm(reader, total=count_lines(data_path)))

    annotator = PipelineAnnotator(annotator_config)
    all_data = {}
    for item in dataset:
        item['answers'] = {answer: {} for answer in item['answers']}
        all_data[item['question']] = item

    answers = sum((list(batch['answers'].keys()) for batch in all_data.values()), [])
    questions = sum(([question] * len(batch['answers']) for question, batch in all_data.items()), [])

    titles = sum(([batch.get('titles', batch.get("ground"))] * len(batch['answers']) for batch in all_data.values()), [])
    materials = None
    if 'materials' in next(iter(all_data.values())):
        materials = sum(([batch['materials']] * len(batch['answers']) for batch in all_data.values()), [])
    extra_materials = None
    if 'extra_materials' in next(iter(all_data.values())):
        extra_materials = sum((
            [batch['extra_materials']] * len(batch['answers'])
            for batch in all_data.values()
        ), [])

    annotations = annotator.annotate(
        answers, questions,
        titles=titles, materials=materials, extra_materials=extra_materials
    )
    for question, answer, annotation in zip(questions, answers, annotations):
        all_data[question]['answers'][answer] = annotation

    all_data = list(all_data.values())
    with jsonlines.open(output_path, 'w', flush=True) as writer:
        writer.write_all(tqdm(all_data))

    print(f"Generate {count_lines(output_path)} lines data.")


if __name__ == '__main__':
    CLI(main)
