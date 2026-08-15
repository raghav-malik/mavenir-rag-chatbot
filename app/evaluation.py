import json
from app.generator import generate


# Test questions with expected answers from 3GPP docs
TEST_SET = [
    {
        "question": "What is network slicing in 5G?",
        "expected_keywords": ["network slice", "S-NSSAI", "PDU session", "RAN", "core network"],
        "expected_source": "23501"
    },
    {
        "question": "What is the purpose of the AMF in 5G?",
        "expected_keywords": ["access", "mobility", "management", "registration", "NAS"],
        "expected_source": "23501"
    },
    {
        "question": "What is a PDU session?",
        "expected_keywords": ["PDU", "session", "data", "UPF", "SMF"],
        "expected_source": "23501"
    },
    {
        "question": "What is the role of the SMF?",
        "expected_keywords": ["session", "management", "PDU", "UPF", "QoS"],
        "expected_source": "23501"
    },
    {
        "question": "What is the function of the UPF in 5G architecture?",
        "expected_keywords": ["user plane", "packet", "routing", "forwarding", "data"],
        "expected_source": "23501"
    },
    {
        "question": "What are the main components of 5G system architecture?",
        "expected_keywords": ["AMF", "SMF", "UPF", "RAN", "core"],
        "expected_source": "23501"
    },
    {
        "question": "How does handover work in NR?",
        "expected_keywords": ["handover", "source", "target", "RRC", "gNB"],
        "expected_source": "38300"
    },
    {
        "question": "What is the NG-RAN architecture?",
        "expected_keywords": ["gNB", "ng-eNB", "NG", "Xn", "RAN"],
        "expected_source": "38300"
    },
    {
        "question": "What is the registration procedure in 5G?",
        "expected_keywords": ["registration", "AMF", "UE", "SUPI", "authentication"],
        "expected_source": "23502"
    },
    {
        "question": "What is the purpose of NSSF in 5G?",
        "expected_keywords": ["slice", "selection", "S-NSSAI", "AMF", "NSSF"],
        "expected_source": "23501"
    }
]

# Questions the docs CANNOT answer — system should abstain
ADVERSARIAL_SET = [
    {
        "question": "What is the maximum bandwidth supported by WiFi 7?",
        "expected_behavior": "abstain",
        "reason": "WiFi is not covered in 3GPP specifications"
    },
    {
        "question": "Compare Mavenir's vRAN solution with Nokia's Cloud RAN.",
        "expected_behavior": "abstain",
        "reason": "Vendor-specific products not in 3GPP specs"
    },
    {
        "question": "What changes were introduced in 3GPP Release 25?",
        "expected_behavior": "abstain",
        "reason": "Release 25 does not exist yet"
    },
    {
        "question": "What is the stock price of Qualcomm?",
        "expected_behavior": "abstain",
        "reason": "Financial data not in 3GPP specs"
    },
    {
        "question": "How does TCP three-way handshake work?",
        "expected_behavior": "abstain",
        "reason": "General networking, not 3GPP-specific"
    },
    {
        "question": "What LLM model does Mavenir use in MavAI OPS?",
        "expected_behavior": "abstain",
        "reason": "Mavenir product details not in 3GPP specs"
    },
    {
        "question": "Should I invest in 5G infrastructure stocks?",
        "expected_behavior": "abstain",
        "reason": "Financial advice not in 3GPP specs"
    },
    {
        "question": "Write Python code to parse a 3GPP specification document.",
        "expected_behavior": "abstain",
        "reason": "Code generation request, not a knowledge query"
    }
]

def evaluate_response(result: dict, test_case: dict) -> dict:
    """
    Evaluate a single response against expected values.
    
    Metrics:
    - keyword_coverage: What percentage of expected keywords appear in the answer
    - source_accuracy: Did the retriever pull from the correct document
    - is_grounded: Did the system cite sources (not hallucinate)
    - answered: Did the system provide an answer (not say "insufficient info")
    """
    answer_lower = result["answer"].lower()
    
    # Keyword coverage: how many expected keywords appear in the answer
    keywords_found = sum(
        1 for kw in test_case["expected_keywords"]
        if kw.lower() in answer_lower
    )
    keyword_coverage = keywords_found / len(test_case["expected_keywords"])
    
    # Source accuracy: did retrieval find the right document
    retrieved_sources = [s["source"] for s in result["sources"]]
    source_hit = any(test_case["expected_source"] in src for src in retrieved_sources)
    
    # Grounding check
    is_grounded = result["confidence"] == "GROUNDED_WITH_CITATIONS"
    
    # Did it actually answer
    answered = result["confidence"] != "NOT_GROUNDED"
    
    return {
        "question": test_case["question"],
        "keyword_coverage": keyword_coverage,
        "source_accuracy": source_hit,
        "is_grounded": is_grounded,
        "answered": answered
    }


def run_evaluation():
    """
    Run the full evaluation suite and print results.
    
    This tests the complete pipeline end-to-end:
    query → retrieval → reranking → generation → evaluation
    """
    print("=" * 70)
    print("EVALUATION: 3GPP RAG Chatbot")
    print("=" * 70)
    
    results = []
    
    for i, test_case in enumerate(TEST_SET, 1):
        print(f"\n[{i}/{len(TEST_SET)}] {test_case['question']}")
        
        result = generate(test_case["question"])
        eval_result = evaluate_response(result, test_case)
        results.append(eval_result)
        
        print(f"  Keyword Coverage: {eval_result['keyword_coverage']:.0%}")
        print(f"  Source Accuracy:   {'✓' if eval_result['source_accuracy'] else '✗'}")
        print(f"  Grounded:         {'✓' if eval_result['is_grounded'] else '✗'}")
        print(f"  Answered:         {'✓' if eval_result['answered'] else '✗'}")
    
    # Aggregate metrics
    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS")
    print("=" * 70)
    
    total = len(results)
    avg_keyword = sum(r["keyword_coverage"] for r in results) / total
    source_acc = sum(1 for r in results if r["source_accuracy"]) / total
    grounding_rate = sum(1 for r in results if r["is_grounded"]) / total
    answer_rate = sum(1 for r in results if r["answered"]) / total
    
    print(f"  Average Keyword Coverage: {avg_keyword:.0%}")
    print(f"  Source Accuracy:          {source_acc:.0%}")
    print(f"  Grounding Rate:           {grounding_rate:.0%}")
    print(f"  Answer Rate:              {answer_rate:.0%}")
    print(f"  Total Questions:          {total}")
    
    # Save results to file
    output = {
        "aggregate": {
            "avg_keyword_coverage": round(avg_keyword, 3),
            "source_accuracy": round(source_acc, 3),
            "grounding_rate": round(grounding_rate, 3),
            "answer_rate": round(answer_rate, 3),
            "total_questions": total
        },
        "individual": results
    }
    
    with open("eval/eval_results.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to eval/eval_results.json")
    return output

def run_adversarial_evaluation():
    """
    Test with questions the docs CANNOT answer.
    The system should abstain (say "I don't have enough information").
    A system that answers these confidently is hallucinating.
    """
    print("\n" + "=" * 70)
    print("ADVERSARIAL EVALUATION: Unanswerable Questions")
    print("=" * 70)
    
    results = []
    
    for i, test_case in enumerate(ADVERSARIAL_SET, 1):
        print(f"\n[{i}/{len(ADVERSARIAL_SET)}] {test_case['question']}")
        
        result = generate(test_case["question"])
        
        # Did the system correctly abstain?
        abstained = result["confidence"] == "NOT_GROUNDED"
        
        results.append({
            "question": test_case["question"],
            "expected": "abstain",
            "actual": "abstained" if abstained else "answered (hallucination risk)",
            "correct": abstained,
            "confidence": result["confidence"],
            "reason": test_case["reason"]
        })
        
        status = "✓ Correctly abstained" if abstained else "✗ Answered (should have abstained)"
        print(f"  {status}")
    
    # Aggregate
    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    abstention_accuracy = correct / total
    
    print(f"\n{'=' * 70}")
    print(f"ADVERSARIAL RESULTS")
    print(f"{'=' * 70}")
    print(f"  Abstention Accuracy: {abstention_accuracy:.0%}")
    print(f"  Correctly Abstained: {correct}/{total}")
    print(f"  False Answers:       {total - correct}/{total}")
    
    # Save
    output = {
        "abstention_accuracy": round(abstention_accuracy, 3),
        "correctly_abstained": correct,
        "false_answers": total - correct,
        "total": total,
        "individual": results
    }
    
    with open("eval/adversarial_results.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to eval/adversarial_results.json")
    return output

if __name__ == "__main__":
    run_evaluation()
    run_adversarial_evaluation()