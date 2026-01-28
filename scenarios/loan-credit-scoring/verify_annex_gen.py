import asyncio
import os
import json
from venturalitica.graph.workflow import create_compliance_graph

async def main():
    print("🚀 Starting Annex IV Verification...")
    
    # Check if any evidence exists
    evidence_dir = ".venturalitica"
    if not os.path.exists(evidence_dir) or not os.listdir(evidence_dir):
        print("❌ Error: .venturalitica/ evidence not found. Run pipeline or monitor first.")
        return

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", default="en", help="Target language for the documentation")
    parser.add_argument("--output", default="annex_iv.md", help="Output filename")
    args = parser.parse_args()

    graph = create_compliance_graph(model_name="mistral")
    state = {
        "project_root": os.getcwd(),
        "bom": {},
        "runtime_meta": {},
        "languages": [args.language], # Wrap single language in list
        "sections": {},
        "revision_count": 0,
        "critic_verdict": "",
        "final_markdown": "",
        "translations": {}
    }
    
    async for event in graph.astream(state):
        for node, data in event.items():
            print(f"  ➡️  Node: {node}")
            
            if node == "compiler":
                final_doc = data["final_markdown"]
                print("\n📄 INTERMEDIATE ENGLISH DOCUMENT:")
                print("--------------------------------------------------")
                print(final_doc[:500] + "...") 
                print("--------------------------------------------------")
                # Save English anyway
                with open("annex_iv_en.md", "w") as f:
                    f.write(final_doc)
            
            if node == "translator":
                translations = data.get("translations", {})
                if args.language in translations:
                    final_doc = translations[args.language]
                    with open(args.output, "w") as f:
                        f.write(final_doc)
                    print(f"✅ FINAL Translated Documentation ({args.language}) saved to {args.output}")

if __name__ == "__main__":
    asyncio.run(main())
