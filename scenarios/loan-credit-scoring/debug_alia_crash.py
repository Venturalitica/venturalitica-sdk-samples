import os
import json
import time
import threading
from venturalitica.graph.nodes import NodeFactory
from venturalitica.graph.state import ComplianceState

def get_nodes():
    # Use provider='transformers' to trigger local ALIA
    return NodeFactory(model_name="alia", provider="transformers")

def worker(nodes, section_id, results):
    print(f"✍️  [Thread {section_id}] Starting generation...")
    # Mock state for the generic generator
    state: ComplianceState = {
        "bom": {"components": []},
        "runtime_meta": {"audit_results": "Test audit"},
        "code_context": {},
        "language": "Spanish",
        "sections": {}
    }
    try:
        # Simulate a bit of wait to overlap if possible
        time.sleep(0.5)
        # Call the internal method that uses the lock
        draft = nodes._generate_generic_section(section_id, state)
        if draft["status"] == "completed":
            results[section_id] = "SUCCESS"
            print(f"✅ [Thread {section_id}] Finished.")
        else:
            results[section_id] = f"ERROR: {draft['content']}"
            print(f"❌ [Thread {section_id}] SDK Error: {draft['content']}")
    except Exception as e:
        results[section_id] = f"CRASH: {str(e)}"
        print(f"🔥 [Thread {section_id}] Process Crash: {str(e)}")

def run_debug_session(parallel=True):
    nodes = get_nodes()
    results = {}
    threads = []
    sections = ["2.a", "2.b", "2.c"]
    
    print(f"\n🚀 Starting SDK Debug Session (Parallel={parallel})\n")
    
    if parallel:
        for s in sections:
            t = threading.Thread(target=worker, args=(nodes, s, results))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
    else:
        for s in sections:
            worker(nodes, s, results)
            
    print("\n📊 Results Summary:")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    run_debug_session(parallel=True)
