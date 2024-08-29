# PDF Sentences Checker with Haystack ( SCH )
Plot the Embedding and your prompt

## Usage
```bash
docker build -t sch \
-f docker/Dockerfile .
```

```bash
docker run -it --rm \
--gpus=all \
-v $(pwd):/ws \
-w /ws \
-p 8501:8501 \
sch \
streamlit run app.py
```