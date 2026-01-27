import asyncio
import os
import argparse
from venturalitica.graph.workflow import create_compliance_graph

LANGUAGES = {
    # EU Official
    "Bulgarian": "bg", "Croatian": "hr", "Czech": "cs", "Danish": "da", 
    "Dutch": "nl", "English": "en", "Estonian": "et", "Finnish": "fi", 
    "French": "fr", "German": "de", "Greek": "el", "Hungarian": "hu", 
    "Irish": "ga", "Italian": "it", "Latvian": "lv", "Lithuanian": "lt", 
    "Maltese": "mt", "Polish": "pl", "Portuguese": "pt", "Romanian": "ro", 
    "Slovak": "sk", "Slovenian": "sl", "Spanish": "es", "Swedish": "sv",
    # Regional / Requested
    "Galician": "gl", "Catalan": "ca", "Basque": "eu", "Asturian": "ast", 
    "Occitan": "oc", "Esperanto": "eo"
}

def resolve_languages(user_input: str) -> dict:
    if not user_input or user_input.lower() == "all":
        return LANGUAGES
    
    selected = {}
    codes = [c.strip().lower() for c in user_input.split(",")]
    # Reverse lookup for simplicity
    code_to_name = {v: k for k, v in LANGUAGES.items()}
    
    for c in codes:
        if c in code_to_name:
            selected[code_to_name[c]] = c
        else:
            # Try to match name
            for name, code in LANGUAGES.items():
                if name.lower() == c:
                    selected[name] = code
                    break
    
    if not selected:
        print(f"⚠️ Warning: No valid languages found for input '{user_input}'. Defaulting to English only.")
        return {"English": "en"}
        
    return selected

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mistral", help="LLM model to use")
    parser.add_argument("--languages", default="en,es", help="Comma-separated language codes (e.g. 'en,es,fr') or 'all'")
    args = parser.parse_args()

    target_languages = resolve_languages(args.languages)
    print(f"🚀 [BATCH] Starting Integrated Multi-lingual Generation ({len(target_languages)} languages: {list(target_languages.values())})")
    
    graph = create_compliance_graph(model_name=args.model)
    
    # Initialize state with all requested languages
    state = {
        "project_root": os.getcwd(),
        "bom": {},
        "runtime_meta": {},
        "languages": list(target_languages.keys()), 
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
                # Master English draft is ready
                master_file = "annex_iv_en.md"
                with open(master_file, "w") as f:
                    f.write(data["final_markdown"])
                print(f"✅ Master English draft saved to {master_file}")
                
            if node == "translator":
                # All translations are ready
                translations = data.get("translations", {})
                for lang_name, doc in translations.items():
                    # Look up code from our filtered list or global list
                    lang_code = LANGUAGES.get(lang_name, "unknown")
                    output_file = f"annex_iv_{lang_code}.md"
                    with open(output_file, "w") as f:
                        f.write(doc)
                    print(f"✅ Translated ({lang_name}) saved to {output_file}")

    print("\n🎉 All requested translations completed from the same master document!")

if __name__ == "__main__":
    asyncio.run(main())
