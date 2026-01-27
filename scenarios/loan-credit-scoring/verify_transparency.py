import os
import json
import asyncio
import os
import json
import asyncio
from venturalitica.graph.nodes import NodeFactory
from venturalitica.graph.state import ComplianceState

def verify_transparency():
    print("🔬 Verifying Transparency Features...")
    
    # 1. Setup Dummy Project Root
    project_root = os.getcwd()
    
    # 2. Instantiate Scanner
    nodes = NodeFactory(model_name="mistral")
    
    # 3. Create Dummy State
    state = {
        "project_root": project_root,
        "bom": {},
        "runtime_meta": {},
        "languages": ["English"],
        "sections": {},
        "evidence_hash": "",
        "bom_security": {}
    }
    
    # 4. Run Scan
    print("  Running ScanProject...")
    result = nodes.scan_project(state)
    
    # 5. Verify Outputs
    evidence_hash = result.get("evidence_hash")
    bom_security = result.get("bom_security")
    
    print("\n📊 Verification Results:")
    
    # Check Hash
    if evidence_hash and len(evidence_hash) == 64:
        print(f"  ✅ Evidence Hash Generated: {evidence_hash[:12]}... (Valid SHA-256)")
    else:
        print(f"  ❌ Invalid Evidence Hash: {evidence_hash}")
        
    # Check Security
    if bom_security is not None:
        print(f"  ✅ Security Object Present")
        if bom_security.get("vulnerable"):
            print(f"  ⚠️ Vulnerabilities Found: {len(bom_security.get('issues', []))}")
        else:
            print(f"  ✅ No Vulnerabilities Found (or BOM empty)")
    else:
        print(f"  ❌ Security Object Missing")

if __name__ == "__main__":
    verify_transparency()
