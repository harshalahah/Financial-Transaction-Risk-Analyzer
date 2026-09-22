.PHONY: install data train serve test demo
install: ; pip install -r requirements.txt
data:    ; python src/generate_data.py --out data/transactions.csv
train:   ; cd src && python train.py
serve:   ; cd src && uvicorn api:app --reload --port 8000
test:    ; python -m pytest tests -q
demo:    ; cd src && python demo.py
all: data train test demo
