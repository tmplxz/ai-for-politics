# AI for Political Education - Supplementary Material (anonymized for reviewers)

#### We asked AI – How is our city changing through municipal politics?
This repository entails a modular AI pipeline as well as results from a broad user survey, allowing to explore how generative AI can best support **political education**. For our work, we have analyzed textual information from election platforms with large language models, generated visualizations of these political agendas, and collected feedback from society via a survey. 

The respective research paper "Characterizing the Pros and Cons of Using AI for Political Education" is under review at FAccT '26. The link to the publicy launched webpage as well as team- and city-specific details were omitted for the sake of double-blind reviewing.

## 🕵️ Results and Analysis

A large part of our paper is focused on our online survey - the responses can be found in [CSV format](survey_results.csv), and the statistical analysis and plotting we performed with a [Pythons script](visualize.py). The outputs of our AI Pipeline can be found in [this subfolder](src/frontend/public/political_content_2025).

## 🚀 Running the AI Pipeline

We implemented a central [main.py](src/main.py) script that allows to perform all the processing in individual steps. The [run_ai.sh](run_ai.sh) script gives an idea of how these steps connect to each other. 

If you want to run our solution locally, you can follow the steps below. Note we used AI models with up to 30B parameters, requiring a certain amount of processing power and VRAM. 

1. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

2. Create a `programs` folder and use it to store the political programs that shall be analyzed.

3. Run the AI pipeline:

    ```bash
    for m in "translate" "summarize" "reason" "translate_results" "generate_images"
    do
        python3 src/main.py --mode $m --input_dir programs --output_dir [your directory]
    done
    ```

## 🌐 Launching the Webpage Locally

The repository includes a Docker setup for running our webpage locally.

1. Make sure [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/) are installed.

2. From the project root, start the containers:

```bash
docker compose up
```

3. Open http://localhost:3000 in your browser.

## 📖 About the Project

This project is experimental and aims at stimulating discourse. **It is NOT a voting recommendation.**

© Authors of "Characterizing the Pros and Cons of Using AI for Political Education
