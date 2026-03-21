#!/usr/bin/env python3
"""Generate messy local business listings via the local LLM for churner testing."""

import json
import sys
from pathlib import Path
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8090/v1", api_key="na")
OUT = Path(__file__).parent.parent / "test_data" / "venues.json"

PROMPT = """Generate exactly 20 messy local business/venue listings as a JSON array.
Each listing should be a raw text blob (like you'd find on a scraped webpage or forum post), NOT structured data.

Requirements:
- Vary the format wildly: some are Yelp-style reviews, some are directory entries, some are social media posts, some are just addresses with notes
- Include duplicates: 3-4 businesses should appear twice with different formatting/info
- Include typos, abbreviations, missing info, conflicting details
- Mix of restaurants, bars, coffee shops, gyms, salons, etc.
- All in the same fictional city "Millbrook Heights"
- Some should have phone numbers (various formats), websites, hours, prices
- Some should be very sparse (just a name and neighborhood)

Output ONLY valid JSON: [{"raw_text": "..."}, {"raw_text": "..."}, ...]"""

def generate_batch(batch_num):
    print(f"Generating batch {batch_num}...")
    resp = client.chat.completions.create(
        model="qwen3-32b",
        messages=[{"role": "user", "content": PROMPT}],
        max_tokens=4096,
        temperature=0.8,
    )
    content = resp.choices[0].message.content or resp.choices[0].message.reasoning_content or ""
    # Extract JSON from response (might have markdown fences)
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0]
    elif "```" in content:
        content = content.split("```")[1].split("```")[0]
    return json.loads(content.strip())

all_records = []
for i in range(5):  # 5 batches × 20 = 100 records
    try:
        batch = generate_batch(i + 1)
        all_records.extend(batch)
        print(f"  Got {len(batch)} records (total: {len(all_records)})")
    except Exception as e:
        print(f"  Batch {i+1} failed: {e}")

OUT.write_text(json.dumps(all_records, indent=2))
print(f"\nWrote {len(all_records)} records to {OUT}")
