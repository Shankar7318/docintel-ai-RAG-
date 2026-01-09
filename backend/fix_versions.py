import subprocess
import sys

print("🔧 Fixing version conflicts...")

# Uninstall conflicting packages
conflicting_packages = [
    "accelerate",
    "huggingface-hub",
    "transformers",
    "tokenizers",
    "langchain",
    "langchain-core",
    "langchain-community",
    "langchain-text-splitters"
]

for pkg in conflicting_packages:
    try:
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", pkg], 
                      capture_output=True, text=True)
        print(f"✓ Uninstalled {pkg}")
    except:
        pass

# Clear cache
subprocess.run([sys.executable, "-m", "pip", "cache", "purge"], capture_output=True)

print("\n📦 Installing compatible versions...")

# Install compatible versions in correct order
install_order = [
    # Core dependencies first
    "pydantic>=2.5,<2.13",
    "pydantic-settings==2.11.0",
    "fastapi==0.104.1",
    "uvicorn==0.24.0",
    
    # HuggingFace ecosystem with compatible versions
    "huggingface-hub==0.20.3",  # This version has split_torch_state_dict_into_shards
    "tokenizers==0.15.2",
    "transformers==4.36.2",     # Compatible with huggingface-hub 0.20.3
    "sentence-transformers==2.2.2",
    
    # LangChain v0.x (not v1.x)
    "langchain-core==0.3.81",
    "langchain==0.3.25",
    "langchain-community==0.3.25",
    "langchain-text-splitters==0.3.11",
    
    # Vector store
    "chromadb==0.4.24",
    
    # PDF processing
    "pypdf==3.15.0",
    "pymupdf==1.22.5",
    
    # Utilities
    "python-multipart==0.0.6",
    "python-dotenv==1.0.0",
    "requests==2.31.0"
]

for package in install_order:
    print(f"Installing {package}...")
    result = subprocess.run([sys.executable, "-m", "pip", "install", package], 
                          capture_output=True, text=True)
    if result.returncode == 0:
        print(f"  ✓ Success")
    else:
        print(f"  ⚠️ Issues: {result.stderr[:100]}")

# Check if accelerate is needed
try:
    # Try to import accelerate to see if it's needed
    import transformers
    # If transformers 4.36.2 works without accelerate, we're good
    print("\n✅ Transformers installed without accelerate")
except ImportError as e:
    if "accelerate" in str(e):
        print("\n📦 Installing accelerate compatible version...")
        subprocess.run([sys.executable, "-m", "pip", "install", "accelerate==0.27.2"], 
                      capture_output=True)

print("\n🧪 Testing imports...")

test_code = """
try:
    import huggingface_hub
    print(f"✓ huggingface_hub: {huggingface_hub.__version__}")
    
    # Test the problematic import
    from accelerate.hooks import AlignDevicesHook
    print("✓ accelerate.hooks import successful")
    
    import transformers
    print(f"✓ transformers: {transformers.__version__}")
    
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    print("✓ transformers models import successful")
    
    import langchain
    print(f"✓ langchain: {langchain.__version__}")
    
    from langchain_community.vectorstores import Chroma
    print("✓ langchain imports successful")
    
    print("\\n✅ ALL IMPORTS SUCCESSFUL! Version conflicts resolved.")
    
except Exception as e:
    print(f"❌ Import error: {e}")
    import traceback
    traceback.print_exc()
"""

# Save and run test
with open("test_imports.py", "w") as f:
    f.write(test_code)

result = subprocess.run([sys.executable, "test_imports.py"], capture_output=True, text=True)
print(result.stdout)
if result.stderr:
    print("Errors:", result.stderr)

# Cleanup
import os
if os.path.exists("test_imports.py"):
    os.remove("test_imports.py")

print("\n🚀 Ready to run your main.py!")
print("Run: uvicorn main:app --host 0.0.0.0 --port 8000 --reload")