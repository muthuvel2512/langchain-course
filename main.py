import os

from dotenv import load_dotenv

load_dotenv()


def main() -> None:
    print(os.environ.get("GROQ_API_KEY"))


if __name__ == "__main__":
    main()
