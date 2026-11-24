from rag import answer_question, get_genai_client, load_index, print_result

DEMO_QUESTIONS = [
    # 5 answerable questions from the scraped Algiers OpenStreetMap dataset
    ("Answerable 1", "Where is Clinique Yamina located?"),
    ("Answerable 2", "What is the phone number and address of بوتوشانت طب الداخلي?"),
    ("Answerable 3", "Which medical facility is located in Birtouta?"),
    ("Answerable 4", "Find Dr LARIBI gynécologue and give its location and coordinates."),
    ("Answerable 5", "Where is عيادة القدس located?"),
    # 2 out-of-scope questions not present in the dataset (gracefully declined)
    ("Out-of-scope 1", "What are the opening hours and consultation fees of Clinique Yamina?"),
    ("Out-of-scope 2", "Can you recommend a cardiologist in Oran, Algeria?"),
]


def main() -> None:
    client = get_genai_client()
    matrix, chunks = load_index()

    print("\n" + "#" * 60)
    print("  RUNNING 7-QUESTION RAG DEMO (5 Answerable + 2 Out-of-Scope)")
    print("#" * 60 + "\n")

    for idx, (label, question) in enumerate(DEMO_QUESTIONS, start=1):
        print("=" * 60)
        print(f"[{label}] Question {idx}/7: {question}")
        answer, results = answer_question(client, question, matrix, chunks)
        print_result(answer, results)


if __name__ == "__main__":
    main()
