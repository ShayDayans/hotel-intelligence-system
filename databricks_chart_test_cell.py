"""
Databricks Test Cell - Verify Charts Appear First in Agent Response
Copy this cell into your Databricks notebook to test the chart fix
"""

import sys
sys.path.insert(0, "/Workspace/Users/alon-gilad@campus.technion.ac.il/hotel-intelligence-system")
sys.path.insert(0, "/Workspace/Users/alon-gilad@campus.technion.ac.il/hotel-intelligence-system/agents")

import builtins
builtins.dbutils = dbutils

from agents.coordinator import LangGraphCoordinator
import agents.databricks_tools as databricks_tools

# ==============================================
# CONFIGURATION
# ==============================================
HOTEL_ID = "ABB_40458495"
HOTEL_NAME = "Rental unit in Broadbeach"  
CITY = "Broadbeach"
QUESTION = "What features should I improve to increase my rating?"

print("=" * 70)
print("🧪 TESTING: Charts First in Agent Response")
print("=" * 70)
print(f"Hotel: {HOTEL_NAME}")
print(f"Question: {QUESTION}")
print("=" * 70)

# Initialize
print("\n[1/3] Initializing coordinator...")
coordinator = LangGraphCoordinator(HOTEL_ID, HOTEL_NAME, CITY)
state = coordinator.get_initial_state()
databricks_tools.clear_chart_urls()
print("✓ Ready")

# Run analysis
print("\n[2/3] Running analysis (15-20 min)...")
print("-" * 70)

response, updated_state = coordinator.run(QUESTION, state)

print("-" * 70)
print("✓ Analysis complete!")

# Verify charts
print("\n[3/3] Verifying chart placement...")
print("=" * 70)

chart_urls = databricks_tools.get_chart_urls()
print(f"\n✓ Found {len(chart_urls)} chart(s) in databricks_tools")

# Check if response starts with charts
response_lines = response.strip().split('\n')
first_meaningful_line = next((line for line in response_lines if line.strip()), "")

if "📊" in response[:200]:
    print("✅ SUCCESS: Charts appear at the beginning of response!")
elif "📊" in response:
    print("⚠️  WARNING: Charts found but not at the beginning")
    chart_position = response.index("📊")
    print(f"   Chart first appears at character {chart_position}/{len(response)}")
else:
    print("❌ ERROR: No charts found in response")

print("\n" + "=" * 70)
print("📄 AGENT RESPONSE:")
print("=" * 70)
print(response)
print("=" * 70)

# Display charts
if chart_urls:
    print("\n🎨 Rendering Charts...")
    for i, url in enumerate(chart_urls, 1):
        chart_name = url.split('/')[-1].replace('.png', '').replace('_', ' ').title()
        displayHTML(f'''
        <div style="margin: 20px 0; padding: 20px; 
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    border-radius: 12px;">
            <h2 style="color: white;">📊 {chart_name}</h2>
            <div style="background: white; padding: 15px; border-radius: 8px; margin-top: 10px;">
                <img src="{url}" style="max-width: 100%; height: auto;"/>
            </div>
            <a href="{url}" target="_blank" 
               style="color: white; background: rgba(255,255,255,0.2); 
                      padding: 8px 16px; border-radius: 5px; text-decoration: none; 
                      display: inline-block; margin-top: 10px;">
                🔗 Open in New Tab
            </a>
        </div>
        ''')

print("\n✅ Test complete!")
