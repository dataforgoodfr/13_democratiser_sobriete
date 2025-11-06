import sys
from pathlib import Path

# Add the rag_system path to sys.path to access taxonomy
current_dir = Path(__file__).parent.resolve()
rag_system_dir = current_dir.parent.parent.parent
sys.path.append(str(rag_system_dir))

print(f"Added to sys.path: {rag_system_dir}")

try:
    from taxonomy.taxonomy.themes_taxonomy import Studied_sector

    print("✅ Successfully imported Studied_sector!")
    print(f"Number of sectors: {len(Studied_sector)}")
    print(f"First 3 sectors: {list(Studied_sector)[:3]}")

    # Test the values() method used in the main script
    print(f"Sample values: {list(Studied_sector.values())[:3]}")
    print(f"Sample keys: {list(Studied_sector.keys())[:3]}")

except ImportError as e:
    print(f"❌ Import failed: {e}")
    print(f"Checked path: {rag_system_dir}")
    print(f"Taxonomy dir exists: {(rag_system_dir / 'taxonomy').exists()}")
    print(
        f"Themes taxonomy file exists: {(rag_system_dir / 'taxonomy' / 'taxonomy' / 'themes_taxonomy.py').exists()}"
    )
