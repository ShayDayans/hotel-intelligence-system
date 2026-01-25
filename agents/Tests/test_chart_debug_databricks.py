"""
Chart Debug Test for Databricks
================================
Run this in a Databricks cell (NOT in Gradio) to debug chart issues.

This script:
1. Calls run_lr_analysis directly
2. Inspects the result JSON structure
3. Checks where charts are stored
4. Tests format_lr_insights output
5. Verifies chart URLs are being generated
"""

import sys
import json

# Setup paths for Databricks
sys.path.insert(0, "/Workspace/Users/alon-gilad@campus.technion.ac.il/hotel-intelligence-system")
sys.path.insert(0, "/Workspace/Users/alon-gilad@campus.technion.ac.il/hotel-intelligence-system/agents")

import builtins
builtins.dbutils = dbutils

# ==============================================
# CONFIGURATION
# ==============================================
HOTEL_ID = "ABB_40458495"
PROPERTY_ID = "40458495"  # Raw ID without prefix

print("=" * 70)
print("🔍 CHART DEBUG TEST")
print("=" * 70)

# ==============================================
# STEP 1: Test direct LR analysis call
# ==============================================
print("\n[STEP 1] Calling run_lr_analysis directly...")
print("-" * 70)

from databricks_tools import (
    run_lr_analysis, 
    format_lr_insights, 
    get_chart_urls,
    clear_chart_urls,
    save_chart_to_filestore,
    extract_raw_id,
    FILESTORE_CHARTS_PATH
)

# Clear any existing chart URLs
clear_chart_urls()
print(f"✓ Cleared previous chart URLs")

# Run the analysis
print(f"\n⏳ Running LR analysis for {HOTEL_ID}...")
print(f"   (This takes ~15 minutes)")
result = run_lr_analysis(HOTEL_ID)

print(f"\n✓ Analysis complete!")
print(f"   Status: {result.get('status')}")

# ==============================================
# STEP 2: Inspect result structure
# ==============================================
print("\n[STEP 2] Inspecting result structure...")
print("-" * 70)

print(f"\n📦 Top-level keys in result:")
for key in result.keys():
    val = result[key]
    if isinstance(val, dict):
        print(f"   • {key}: dict with {len(val)} keys -> {list(val.keys())[:5]}...")
    elif isinstance(val, list):
        print(f"   • {key}: list with {len(val)} items")
    elif isinstance(val, str) and len(val) > 100:
        print(f"   • {key}: string ({len(val)} chars)")
    else:
        print(f"   • {key}: {type(val).__name__} = {str(val)[:50]}")

# ==============================================
# STEP 3: Find charts in result
# ==============================================
print("\n[STEP 3] Searching for charts in result...")
print("-" * 70)

# Check direct 'charts' key
charts_direct = result.get("charts", {})
print(f"\n1. result['charts']:")
if charts_direct:
    print(f"   ✓ FOUND! Keys: {list(charts_direct.keys())}")
    for k, v in charts_direct.items():
        if isinstance(v, str):
            print(f"      • {k}: base64 string ({len(v)} chars)")
        elif isinstance(v, dict):
            print(f"      • {k}: dict with keys {list(v.keys())}")
else:
    print(f"   ✗ NOT FOUND (empty or missing)")

# Check ui_artifacts.charts
ui_artifacts = result.get("ui_artifacts", {})
print(f"\n2. result['ui_artifacts']:")
if ui_artifacts:
    print(f"   ✓ FOUND! Type: {type(ui_artifacts).__name__}")
    if isinstance(ui_artifacts, dict):
        print(f"   Keys: {list(ui_artifacts.keys())}")
        charts_ui = ui_artifacts.get("charts", {})
        if charts_ui:
            print(f"   ui_artifacts['charts'] keys: {list(charts_ui.keys())}")
            for k, v in charts_ui.items():
                if isinstance(v, str):
                    print(f"      • {k}: base64 string ({len(v)} chars)")
                elif isinstance(v, dict):
                    print(f"      • {k}: dict with keys {list(v.keys())}")
        else:
            print(f"   ui_artifacts['charts']: NOT FOUND")
else:
    print(f"   ✗ NOT FOUND")

# Check llm_context for charts
llm_context = result.get("llm_context", {})
print(f"\n3. result['llm_context']:")
if llm_context:
    print(f"   ✓ FOUND! Keys: {list(llm_context.keys())}")
    if "charts" in llm_context:
        print(f"   llm_context['charts']: {list(llm_context['charts'].keys())}")
else:
    print(f"   ✗ NOT FOUND")

# Deep search for any key containing 'chart'
print(f"\n4. Deep search for 'chart' in any key...")
def find_chart_keys(obj, path="result"):
    found = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if "chart" in k.lower():
                found.append(f"{path}['{k}']")
            found.extend(find_chart_keys(v, f"{path}['{k}']"))
    elif isinstance(obj, list):
        for i, item in enumerate(obj[:3]):  # Check first 3 items only
            found.extend(find_chart_keys(item, f"{path}[{i}]"))
    return found

chart_paths = find_chart_keys(result)
if chart_paths:
    print(f"   ✓ Found chart-related keys:")
    for p in chart_paths:
        print(f"      • {p}")
else:
    print(f"   ✗ No chart-related keys found anywhere")

# ==============================================
# STEP 4: Test format_lr_insights
# ==============================================
print("\n[STEP 4] Testing format_lr_insights...")
print("-" * 70)

formatted_output = format_lr_insights(result, top_n=5, property_id=HOTEL_ID)

print(f"\n📝 Formatted output length: {len(formatted_output)} chars")
print(f"\n📝 First 500 chars of formatted output:")
print("-" * 40)
print(formatted_output[:500])
print("-" * 40)

# Check if charts are in the formatted output
if "📊" in formatted_output:
    print(f"\n✅ Charts marker (📊) found in formatted output!")
    # Find the chart lines
    for line in formatted_output.split('\n'):
        if "📊" in line:
            print(f"   {line}")
else:
    print(f"\n❌ No chart markers found in formatted output!")

# ==============================================
# STEP 5: Check saved chart URLs
# ==============================================
print("\n[STEP 5] Checking saved chart URLs...")
print("-" * 70)

chart_urls = get_chart_urls()
print(f"\n📊 Chart URLs from databricks_tools._last_chart_urls:")
if chart_urls:
    print(f"   ✓ Found {len(chart_urls)} URL(s):")
    for url in chart_urls:
        print(f"      • {url}")
else:
    print(f"   ✗ No chart URLs saved!")

# ==============================================
# STEP 6: Check FileStore for saved files
# ==============================================
print("\n[STEP 6] Checking FileStore directory...")
print("-" * 70)

try:
    file_path = f"{FILESTORE_CHARTS_PATH}/{PROPERTY_ID}/"
    print(f"\nChecking: {file_path}")
    files = dbutils.fs.ls(file_path)
    print(f"   ✓ Found {len(files)} file(s):")
    for f in files:
        print(f"      • {f.name} ({f.size} bytes)")
        
    # Construct URLs
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.getOrCreate()
    workspace_url = spark.conf.get("spark.databricks.workspaceUrl")
    print(f"\n   URLs would be:")
    for f in files:
        url = f"https://{workspace_url}/files/hotel_intel_charts/{PROPERTY_ID}/{f.name}"
        print(f"      • {url}")
except Exception as e:
    print(f"   ✗ Error checking FileStore: {e}")

# ==============================================
# STEP 7: Manual chart save test
# ==============================================
print("\n[STEP 7] Testing manual chart save...")
print("-" * 70)

# Find actual chart data
chart_data = None
chart_name = None

# Try different locations
if charts_direct:
    chart_name = list(charts_direct.keys())[0]
    chart_data = charts_direct[chart_name]
elif ui_artifacts and isinstance(ui_artifacts, dict):
    charts_ui = ui_artifacts.get("charts", {})
    if charts_ui:
        chart_name = list(charts_ui.keys())[0]
        chart_data = charts_ui[chart_name]

if chart_data:
    print(f"\n📊 Found chart data for '{chart_name}'")
    if isinstance(chart_data, dict):
        print(f"   Type: dict with keys {list(chart_data.keys())}")
        # Extract actual base64 data
        chart_data = chart_data.get("data") or chart_data.get("base64") or chart_data.get("image")
    
    if chart_data and isinstance(chart_data, str):
        print(f"   Base64 length: {len(chart_data)} chars")
        
        # Try to save it manually
        print(f"\n   Attempting manual save...")
        url = save_chart_to_filestore(chart_data, "test_chart", PROPERTY_ID)
        if url:
            print(f"   ✅ Saved successfully: {url}")
        else:
            print(f"   ❌ Save failed!")
    else:
        print(f"   ❌ Could not extract base64 data from chart_data")
else:
    print(f"\n❌ No chart data found to test!")

# ==============================================
# SUMMARY
# ==============================================
print("\n" + "=" * 70)
print("📋 SUMMARY")
print("=" * 70)

issues = []
if not charts_direct and not (ui_artifacts and ui_artifacts.get("charts")):
    issues.append("Charts not found in result JSON")
if "📊" not in formatted_output:
    issues.append("Charts not in formatted output")
if not chart_urls:
    issues.append("Chart URLs not being saved")

if issues:
    print("\n❌ ISSUES FOUND:")
    for i, issue in enumerate(issues, 1):
        print(f"   {i}. {issue}")
    print("\n💡 LIKELY CAUSE: Charts are not in expected location in LR notebook output")
    print("   Check the LR notebook to see where it stores chart data")
else:
    print("\n✅ All checks passed! Charts should be appearing.")

print("\n" + "=" * 70)
print("🔍 DEBUG COMPLETE")
print("=" * 70)

# Print full result structure for manual inspection
print("\n[BONUS] Full result JSON (for manual inspection):")
print("-" * 70)
try:
    # Try to print a clean version without huge base64 strings
    clean_result = {}
    for k, v in result.items():
        if isinstance(v, str) and len(v) > 200:
            clean_result[k] = f"<string: {len(v)} chars>"
        elif isinstance(v, dict):
            clean_dict = {}
            for k2, v2 in v.items():
                if isinstance(v2, str) and len(v2) > 200:
                    clean_dict[k2] = f"<string: {len(v2)} chars>"
                elif isinstance(v2, dict):
                    clean_dict[k2] = {k3: f"<{type(v3).__name__}>" for k3, v3 in v2.items()}
                else:
                    clean_dict[k2] = v2
            clean_result[k] = clean_dict
        else:
            clean_result[k] = v
    print(json.dumps(clean_result, indent=2, default=str))
except Exception as e:
    print(f"Could not print clean result: {e}")
