import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from venturalitica.graph.workflow import create_compliance_graph
from venturalitica.graph.state import ComplianceState

# Setup environment
load_dotenv()

def generate_report(project_root: str = ".", lang: str = "es"):
    """
    Executes the Venturalítica Annex IV generation workflow.
    """
    print(f"🚀 Starting Annex IV Generation (Language: {lang})")
    print(f"📂 Project Root: {os.path.abspath(project_root)}")
    
    # 1. Initialize Graph
    # We use provider='transformers' to trigger the local ALIA model
    app = create_compliance_graph(model_name="alia", provider="transformers")
    
    # 2. Initial State
    initial_state: ComplianceState = {
        "project_root": os.path.abspath(project_root),
        "language": lang,
        "bom": {},
        "runtime_meta": {},
        "sections": {},
        "evidence_hash": "",
        "bom_security": {},
        "code_context": {},
        "final_markdown": "",
        "revision_count": 0,
        "critic_verdict": "",
        "translations": {}
    }
    
    # 3. Run Workflow
    print("⏳ Running compliance graph (this may take several minutes with local ALIA)...")
    try:
        final_state = app.invoke(initial_state)
        
        # 4. Save Results
        output_dir = Path(project_root) / "governance"
        output_dir.mkdir(exist_ok=True)
        
        output_path = output_dir / "annex_iv_technical_doc.md"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(final_state["final_markdown"])
            
        print(f"\n✅ Generation Complete!")
        print(f"📄 Report saved to: {output_path}")
        
    except Exception as e:
        print(f"\n❌ Error during generation: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # You can customize the language here (e.g., 'en' for English)
    target_language = os.getenv("VENTURALITICA_LANG", "es")
    generate_report(lang=target_language)
