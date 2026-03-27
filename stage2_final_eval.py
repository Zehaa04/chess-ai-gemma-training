import os
import json
import torch
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoProcessor, AutoModelForCausalLM
from peft import PeftMixedModel  
import pandas as pd
import re

os.environ["HUGGINGFACE_HUB_TOKEN"] = ""
HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN")
os.environ["GEMMA_DISABLE_PAN_AND_SCAN"] = "1"

ROOT = "DatasetFinal"
TEST_FILE = os.path.join(ROOT, "splits/stage2_llm/test.jsonl")
OUTPUT_DIR = "kokastage2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_NAME = "google/gemma-3-4b-it"
STAGE1_ADAPTER = "stage1_vision_lora_adapter"
STAGE2_ADAPTER = "stage2_llm_lora_adapter"

print("="*70)
print("STAGE 2 EVALUATION - BALANCED (1%)")
print("="*70)
print(f"Base Model: {MODEL_NAME}")
print(f"Stage 1 (Vision): {STAGE1_ADAPTER}/")
print(f"Stage 2 (LLM): {STAGE2_ADAPTER}/")
print(f"Test file: {TEST_FILE}")
print(f"Output: {OUTPUT_DIR}/")
print("="*70)


#=======================================================================
# HELPER FUNCTIONS - BALANCED VERSION
# ============================================================================

def normalize_text(text):
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def extract_number(text):
    """Extract first number from text"""
    match = re.search(r'\d+', text)
    return int(match.group()) if match else None


def extract_chess_move(text):
    """Extract chess move notation"""
    pattern = r"\b([KQRBN]?[a-h]?[1-8]?x?[a-h][1-8][+#]?|O-O-O|O-O)\b"
    matches = re.findall(pattern, text)
    return matches[0] if matches else None


def extract_piece_sequence(text):
    """Extract 8-piece sequence from verbose output"""
    text_norm = normalize_text(text)
    pieces = []
    lines = text.split('\n')
    
    # Check first line for space-separated format
    first_line = lines[0].strip()
    tokens = re.split(r'[\s]+', first_line)
    
    for token in tokens:
        if len(pieces) >= 8:
            break
        token_clean = token.replace("'s", "").replace("(", "").replace(")", "").strip("-").strip(",")
        
        if token_clean == "empty":
            pieces.append("empty")
        elif re.match(r'(white|black)-(pawn|knight|bishop|rook|queen|king)', token_clean):
            pieces.append(token_clean)
    
    if len(pieces) == 8:
        return pieces
    
    # Try extracting from verbose format
    pieces = []
    for line in lines:
        matches = re.findall(r'[a-h][1-8]:\s*([^\n,]+)', line)
        for match in matches:
            if len(pieces) >= 8:
                break
            match_clean = match.strip().lower()
            if match_clean == "empty":
                pieces.append("empty")
            elif any(p in match_clean for p in ["pawn", "knight", "bishop", "rook", "queen", "king"]):
                color = "white" if "white" in match_clean else "black" if "black" in match_clean else None
                piece_type = None
                for p in ["pawn", "knight", "bishop", "rook", "queen", "king"]:
                    if p in match_clean:
                        piece_type = p
                        break
                if color and piece_type:
                    pieces.append(f"{color}-{piece_type}")
    
    return pieces if len(pieces) == 8 else []


def extract_square_piece(text):
    """Extract piece from square identification answer"""
    text_norm = normalize_text(text)
    
    # Check for "empty" first
    if any(word in text_norm for word in ["empty", "no piece", "unoccupied"]):
        return "empty"
    
    # Try to find color + piece pattern
    match = re.search(r'(white|black)[\s\-\']*s?\s*(pawn|knight|bishop|rook|queen|king)', text_norm)
    if match:
        return f"{match.group(1)}-{match.group(2)}"
    
    # Just piece name with color somewhere
    for piece in ["king", "queen", "rook", "bishop", "knight", "pawn"]:
        if piece in text_norm:
            color = "white" if "white" in text_norm else "black" if "black" in text_norm else None
            if color:
                return f"{color}-{piece}"
    
    return text_norm.split('.')[0].split('\n')[0].strip()


def calculate_token_f1(pred, gold):
    pred_tokens = normalize_text(pred).split()
    gold_tokens = normalize_text(gold).split()
    
    if not gold_tokens:
        return 1.0 if not pred_tokens else 0.0
    if not pred_tokens:
        return 0.0
    
    pred_set = set(pred_tokens)
    gold_set = set(gold_tokens)
    common = pred_set & gold_set
    
    precision = len(common) / len(pred_set) if pred_set else 0.0
    recall = len(common) / len(gold_set) if gold_set else 0.0
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * (precision * recall) / (precision + recall)


def is_correct_answer(pred, gold, question_type):
    """BALANCED matching - reasonable but fair"""
    pred_norm = normalize_text(pred)
    gold_norm = normalize_text(gold)
    
    # ========================================================================
    # GAME OUTCOME - Who won the game
    # ========================================================================
    if question_type == "game_outcome":
        white_phrases = ["white", "white won", "white wins", "1-0"]
        black_phrases = ["black", "black won", "black wins", "0-1"]
        draw_phrases = ["draw", "drawn", "it's a draw", "the game was drawn", "1/2-1/2"]
        
        gold_first = gold_norm.split('.')[0].split('\n')[0]
        gold_is_white = any(p in gold_first for p in white_phrases) and not any(p in gold_first for p in black_phrases)
        gold_is_black = any(p in gold_first for p in black_phrases) and not any(p in gold_first for p in white_phrases)
        gold_is_draw = any(p in gold_first for p in draw_phrases)
        
        pred_first = pred_norm.split('.')[0].split('\n')[0]
        pred_is_white = any(p in pred_first for p in white_phrases) and not any(p in pred_first for p in black_phrases)
        pred_is_black = any(p in pred_first for p in black_phrases) and not any(p in pred_first for p in white_phrases)
        pred_is_draw = any(p in pred_first for p in draw_phrases)
        
        if gold_is_white:
            return pred_is_white
        elif gold_is_black:
            return pred_is_black
        elif gold_is_draw:
            return pred_is_draw
        
        return False
    
    # ========================================================================
    # DESCRIBE RANK/FILE/DIAGONAL - Require 6/8 correct (75%)
    # ========================================================================
    if question_type in ["describe_rank", "describe_file", "describe_diagonal"]:
        gold_first_line = gold.split('\n')[0].strip()
        gold_tokens = gold_first_line.split()
        
        pred_seq = extract_piece_sequence(pred)
        
        if pred_seq and len(gold_tokens) == 8:
            gold_seq = []
            for token in gold_tokens:
                token_clean = token.replace("'s", "").replace("(", "").replace(")", "").strip("-").strip(",")
                gold_seq.append(token_clean)
            
            if len(gold_seq) == 8 and len(pred_seq) == 8:
                matches = sum(1 for p, g in zip(pred_seq, gold_seq) if p == g)
                # BALANCED: 6/8 = 75% - allows for 2 mistakes
                return matches >= 6
        
        return False
    
    # ========================================================================
    # SQUARE IDENTIFICATION - Must match BOTH color AND piece
    # ========================================================================
    if question_type == "square_identification":
        pred_piece = extract_square_piece(pred)
        gold_piece = extract_square_piece(gold)
        
        pred_clean = normalize_text(pred_piece)
        gold_clean = normalize_text(gold_piece)
        
        # Empty square
        if gold_clean == "empty" or "empty" in gold_clean:
            return pred_clean == "empty" or "empty" in pred_clean
        
        # Parse gold
        if "-" in gold_piece:
            gold_parts = gold_piece.split("-")
            gold_color = gold_parts[0]
            gold_type = gold_parts[1]
        else:
            gold_color = "white" if "white" in gold_clean else "black" if "black" in gold_clean else None
            gold_type = None
            for piece in ["king", "queen", "rook", "bishop", "knight", "pawn"]:
                if piece in gold_clean:
                    gold_type = piece
                    break
        
        # Parse prediction
        if "-" in pred_piece:
            pred_parts = pred_piece.split("-")
            pred_color = pred_parts[0]
            pred_type = pred_parts[1]
        else:
            pred_color = "white" if "white" in pred_clean else "black" if "black" in pred_clean else None
            pred_type = None
            for piece in ["king", "queen", "rook", "bishop", "knight", "pawn"]:
                if piece in pred_clean:
                    pred_type = piece
                    break
        
        # BALANCED: Must match BOTH color AND piece type
        if gold_color and gold_type and pred_color and pred_type:
            return (gold_color == pred_color) and (gold_type == pred_type)
        
        return False
    
    # ========================================================================
    # BEST MOVE - Extract move notation
    # ========================================================================
    if question_type == "best_move":
        pred_move = extract_chess_move(pred)
        gold_move = extract_chess_move(gold)
        
        if pred_move and gold_move:
            return pred_move.lower() == gold_move.lower()
        
        # Fallback: if gold move appears in prediction
        if gold_move and gold_move.lower() in pred_norm:
            return True
        
        return False
    
    # ========================================================================
    # CHECK DETECTION - Look for yes/no
    # ========================================================================
    if question_type == "check_detection":
        no_phrases = ["no", "not in check", "safe", "neither", "both kings are safe", "no check"]
        yes_phrases = ["yes", "check", "in check", "under attack", "one king"]
        
        # Look at first sentence of gold
        gold_first = gold_norm.split('.')[0].split('\n')[0]
        gold_is_no = any(p in gold_first for p in no_phrases)
        gold_is_yes = any(p in gold_first for p in yes_phrases) and not gold_is_no
        
        # Look at first sentence of prediction
        pred_first = pred_norm.split('.')[0].split('\n')[0]
        pred_is_no = any(p in pred_first for p in no_phrases)
        pred_is_yes = any(p in pred_first for p in yes_phrases) and not pred_is_no
        
        if gold_is_no:
            return pred_is_no
        elif gold_is_yes:
            return pred_is_yes
        
        return False
    
    # ========================================================================
    # MATERIAL EVALUATION - Check for white/black/equal
    # ========================================================================
    if question_type == "material_eval":
        white_phrases = ["white is ahead", "white has more", "white leads", "white advantage"]
        black_phrases = ["black is ahead", "black has more", "black leads", "black advantage"]
        equal_phrases = ["material is equal", "equal", "same", "balanced"]
        
        gold_first = gold_norm.split('.')[0].split('\n')[0]
        gold_is_white = any(p in gold_first for p in white_phrases)
        gold_is_black = any(p in gold_first for p in black_phrases)
        gold_is_equal = any(p in gold_first for p in equal_phrases)
        
        pred_first = pred_norm.split('.')[0].split('\n')[0]
        pred_is_white = any(p in pred_first for p in white_phrases)
        pred_is_black = any(p in pred_first for p in black_phrases)
        pred_is_equal = any(p in pred_first for p in equal_phrases)
        
        if gold_is_white:
            return pred_is_white
        elif gold_is_black:
            return pred_is_black
        elif gold_is_equal:
            return pred_is_equal
        
        return False
    
    # ========================================================================
    # TACTICAL VISION - Must identify at least ONE correct piece
    # ========================================================================
    if question_type == "tactical_vision":
        none_phrases = ["none", "no pieces", "cannot attack", "no black pieces", "no white pieces"]
        
        # Check if gold says "none"
        gold_is_none = any(p in gold_norm for p in none_phrases)
        pred_is_none = any(p in pred_norm for p in none_phrases)
        
        if gold_is_none:
            return pred_is_none
        
        # Extract pieces from gold (e.g., "white-pawn on h4")
        gold_pieces = set()
        gold_matches = re.findall(r'(white|black)-(pawn|knight|bishop|rook|queen|king)', gold_norm)
        for color, piece in gold_matches:
            gold_pieces.add(f"{color}-{piece}")
        
        # Extract from prediction
        pred_pieces = set()
        pred_matches = re.findall(r'(white|black)[\s\-]*(pawn|knight|bishop|rook|queen|king)', pred_norm)
        for color, piece in pred_matches:
            pred_pieces.add(f"{color}-{piece}")
        
        if gold_pieces:
            # BALANCED: Must match at least ONE piece correctly
            overlap = gold_pieces & pred_pieces
            return len(overlap) > 0
        
        return False
    
    # ========================================================================
    # COUNT - Exact match required
    # ========================================================================
    if question_type == "count":
        pred_num = extract_number(pred)
        gold_num = extract_number(gold)
        
        if pred_num is not None and gold_num is not None:
            return pred_num == gold_num
        
        return False
    
    # ========================================================================
    # YES/NO - Simple detection
    # ========================================================================
    if question_type == "yes_no":
        yes_phrases = ["yes"]
        no_phrases = ["no", "empty", "not occupied"]
        
        gold_first = gold_norm.split('.')[0].split('\n')[0]
        gold_is_yes = any(p in gold_first for p in yes_phrases) and not any(p in gold_first for p in no_phrases)
        gold_is_no = any(p in gold_first for p in no_phrases)
        
        pred_first = pred_norm.split('.')[0].split('\n')[0]
        pred_is_yes = any(p in pred_first for p in yes_phrases) and not any(p in pred_first for p in no_phrases)
        pred_is_no = any(p in pred_first for p in no_phrases)
        
        if gold_is_yes:
            return pred_is_yes
        elif gold_is_no:
            return pred_is_no
        
        return False
    
    # Fallback
    return gold_norm in pred_norm or pred_norm in gold_norm


def classify_question_type(question):
    q = question.lower()
    
    # Check for game outcome first
    if any(k in q for k in ["who won", "what was the result", "winner", "outcome of this game", "what is the outcome"]):
        return "game_outcome"
    
    if any(k in q for k in ["best move", "what should", "move for", "to move"]):
        return "best_move"
    if any(k in q for k in ["check", "under attack", "king safe", "is there a check", "any king under attack", "is any king", "is either"]):
        return "check_detection"
    if any(k in q for k in ["material", "ahead", "who has more", "balance"]):
        return "material_eval"
    
    # MOVE THIS BLOCK UP - Check for sequence questions BEFORE tactical_vision
    is_sequence = any(k in q for k in ["list", "state of", "describe the pieces", "analyzing", "identify each", "8 values", "8 space-separated", "current state"])
    
    if is_sequence:
        if "diagonal" in q:
            return "describe_diagonal"
        if re.search(r'file\s+[a-h]', q) or "column" in q:
            return "describe_file"
        if re.search(r'rank\s+[1-8]', q) or "row" in q:
            return "describe_rank"
    
    # NOW check tactical_vision (after sequence checks)
    if any(k in q for k in ["attack", "reach", "can hit", "range", "attacking", "which pieces", "what pieces"]):
        return "tactical_vision"
    
    if any(k in q for k in ["how many", "count", "number of"]):
        return "count"

    if any(k in q for k in ["square", "identify", "locate", "what piece", "occupies", "is at", "what is on", "is located"]):
        return "square_identification"
    
    return "other"



def run_inference_single(image_obj, question):
    q_lower = question.lower()
    needs_long = any(phrase in q_lower for phrase in ["analyzing rank", "analyzing file", "describe", "current state"])
    
    if needs_long:
        max_tokens = 150
        full_question = question.strip()
    else:
        max_tokens = 32
        concise_suffix = "\n\nAnswer concisely in 1-2 words or a short phrase. Do not explain."
        full_question = question.strip() + concise_suffix
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_obj},
                {"type": "text", "text": full_question}
            ]
        }
    ]
    
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )
    
    input_len = inputs["input_ids"].shape[-1]
    gen_ids = outputs[0][input_len:]
    pred = processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    
    return pred


# ============================================================================
# LOAD MODEL
# ============================================================================
print("\n" + "="*70)
print("LOADING MODEL WITH BOTH ADAPTERS")
print("="*70)

processor = AutoProcessor.from_pretrained(
    STAGE2_ADAPTER,
    token=HF_TOKEN,
    trust_remote_code=True,
)

print("Loading base model in bfloat16...")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    token=HF_TOKEN,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

model = PeftMixedModel.from_pretrained(base_model, STAGE1_ADAPTER, adapter_name="stage1_vision")
model.load_adapter(STAGE2_ADAPTER, adapter_name="stage2_llm")
model.set_adapter(["stage1_vision", "stage2_llm"])
model.eval()
torch.set_grad_enabled(False)

print(f"\n✅ Model loaded with both adapters active")
print("="*70)

# ============================================================================
# LOAD DATA
# ============================================================================
with open(TEST_FILE, 'r', encoding='utf-8') as f:
    all_data = [json.loads(line) for line in f]

subset_size = max(1, int(0.15 * len(all_data)))
test_data = all_data[:subset_size]

print(f"\n✅ Loaded {len(all_data):,} total samples")
print(f"✅ Evaluating {len(test_data):,} samples (10% for testing)")

# ============================================================================
# RUN EVALUATION
# ============================================================================
grouped_data = defaultdict(list)
for item in test_data:
    img_path = item["messages"][0]["content"][0]["image"]
    grouped_data[img_path].append(item)

results = []
question_type_stats = defaultdict(lambda: {
    "total": 0,
    "correct": 0,
    "f1_scores": [],
    "mae_errors": []
})

print(f"\nProcessing {len(grouped_data)} unique images...")
for img_path, items in tqdm(grouped_data.items(), desc="Evaluating"):
    full_path = os.path.join(ROOT, img_path)
    if not os.path.exists(full_path):
        print(f"Warning: {full_path} not found")
        continue
    
    image_obj = Image.open(full_path).convert("RGB")
    
    questions = [it["messages"][0]["content"][1]["text"] for it in items]
    ground_truths = [it["messages"][1]["content"][0]["text"] for it in items]
    
    predictions = []
    for question in questions:
        pred = run_inference_single(image_obj, question)
        predictions.append(pred)
    
    for prediction, ground_truth, question in zip(predictions, ground_truths, questions):
        q_type = classify_question_type(question)
        
        is_correct = is_correct_answer(prediction, ground_truth, q_type)
        exact_match = int(is_correct)
        token_f1 = calculate_token_f1(prediction, ground_truth)
        
        stats = question_type_stats[q_type]
        stats["total"] += 1
        stats["correct"] += exact_match
        stats["f1_scores"].append(token_f1)
        
        if q_type == "count":
            p_num = extract_number(prediction)
            g_num = extract_number(ground_truth)
            if p_num is not None and g_num is not None:
                stats["mae_errors"].append(abs(p_num - g_num))
        
        results.append({
            "image": img_path,
            "question": question,
            "ground_truth": ground_truth,
            "prediction": prediction,
            "question_type": q_type,
            "exact_match": exact_match,
            "token_f1": token_f1
        })

print(f"✅ Evaluated {len(results)} samples")

# ============================================================================
# PRINT RESULTS
# ============================================================================
total_samples = len(results)
total_correct = sum(r["exact_match"] for r in results)
overall_accuracy = total_correct / total_samples * 100
overall_f1 = sum(r["token_f1"] for r in results) / total_samples * 100

print(f"\n{'='*70}")
print(f"STAGE 2 FINETUNED RESULTS (BALANCED EVALUATION)")
print(f"{'='*70}")
print(f"Accuracy: {overall_accuracy:.2f}% ({total_correct}/{total_samples})")
print(f"Token F1: {overall_f1:.2f}%")
print()

type_summary = []
for q_type, stats in sorted(question_type_stats.items()):
    acc = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
    avg_f1 = sum(stats["f1_scores"]) / len(stats["f1_scores"]) * 100 if stats["f1_scores"] else 0
    
    type_row = {
        "Question Type": q_type,
        "Total": stats["total"],
        "Correct": stats["correct"],
        "Accuracy (%)": f"{acc:.2f}",
        "Avg F1 (%)": f"{avg_f1:.2f}",
    }
    
    if q_type == "count" and stats["mae_errors"]:
        mae = sum(stats["mae_errors"]) / len(stats["mae_errors"])
        type_row["MAE"] = f"{mae:.2f}"
    
    type_summary.append(type_row)
    
    print(f"{q_type:25s} | Acc: {acc:6.2f}% ({stats['correct']}/{stats['total']})")

print(f"\n{'='*70}")
print("SAMPLE PREDICTIONS (3 per type)")
print(f"{'='*70}")

samples_by_type = defaultdict(list)
for r in results:
    samples_by_type[r['question_type']].append(r)

count = 0
for q_type in sorted(samples_by_type.keys()):
    print(f"\n--- {q_type.upper()} ---")
    for r in samples_by_type[q_type][:3]:
        count += 1
        symbol = "✓" if r["exact_match"] else "✗"
        print(f"\n{count}. Q: {r['question'][:80]}...")
        print(f"   GT:   {r['ground_truth']}")
        print(f"   PRED: {r['prediction'][:100]}... {symbol}")

# ============================================================================
# SAVE RESULTS
# ============================================================================
predictions_file = os.path.join(OUTPUT_DIR, "predictions.jsonl")
with open(predictions_file, 'w', encoding='utf-8') as f:
    for r in results:
        f.write(json.dumps(r) + '\n')

summary_file = os.path.join(OUTPUT_DIR, "summary.csv")
pd.DataFrame(type_summary).to_csv(summary_file, index=False)

metrics_file = os.path.join(OUTPUT_DIR, "metrics.json")
with open(metrics_file, 'w') as f:
    json.dump({
        "dataset": "stage2_llm",
        "model": "Stage 1 (vision) + Stage 2 (LLM) adapters",
        "stage1_adapter": STAGE1_ADAPTER,
        "stage2_adapter": STAGE2_ADAPTER,
        "total_samples": total_samples,
        "exact_match_accuracy": overall_accuracy,
        "token_f1": overall_f1,
        "correct": total_correct,
    }, f, indent=2)

print(f"\n✅ Results saved to {OUTPUT_DIR}/")
print(f"   - predictions.jsonl")
print(f"   - summary.csv")
print(f"   - metrics.json")

print(f"\n{'='*70}")
print(f"STAGE 2 FINETUNED: {overall_accuracy:.2f}% accuracy")
print(f"{'='*70}")