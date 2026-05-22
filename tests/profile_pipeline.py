import requests
import numpy as np
import time

API_URL = "http://localhost:8000/upload"
TEST_IMAGE = "data/samples/11318453.tif"
ITERATIONS = 20 

def run_detailed_benchmark():
    stats = {
        "vision": [],
        "ocr": [],
        "ingestion": [],
        "total": []
    }

    print(f"Profiling {ITERATIONS} documents...")

    for i in range(ITERATIONS):
        with open(TEST_IMAGE, "rb") as f:
            r = requests.post(API_URL, files={"file": f})
        
        if r.status_code == 200:
            # Gather the latency dictionary from json
            data = r.json()["latencies"]
            stats["vision"].append(data["vision_ms"])
            stats["ocr"].append(data["ocr_ms"])
            stats["ingestion"].append(data["ingestion_ms"])
            stats["total"].append(data["total_ms"])
            
            print(f"Iteration {i+1}/{ITERATIONS} done", end="\r")
        else:
            print(f"\nError at iteration {i+1}: {r.status_code}")

    print("\n\n" + "="*45)
    print("DETAILED LATENCY BREAKDOWN (Averages)")
    print("="*45)
    print(f"1. Vision (EfficientNet ONNX) : {np.mean(stats['vision']):.2f} ms")
    print(f"2. OCR (EasyOCR on CPU) : {np.mean(stats['ocr'])/1000:.2f} s")
    print(f"3. RAG Ingestion (FAISS): {np.mean(stats['ingestion']):.2f} ms")
    print("-" * 45)
    print(f"TOTAL PIPELINE LATENCY: {np.mean(stats['total'])/1000:.2f} s")
    print(f"THROUGHPUT: {60 / (np.mean(stats['total'])/1000):.2f} docs/min")
    print("="*45)

if __name__ == "__main__":
    run_detailed_benchmark()