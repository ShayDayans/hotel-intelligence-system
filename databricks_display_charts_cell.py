"""
Quick Chart Display Cell for Databricks
Run this after any analysis that generates charts
"""

import agents.databricks_tools as databricks_tools

# Get chart URLs
chart_urls = databricks_tools.get_chart_urls()

if chart_urls:
    print(f"✓ Found {len(chart_urls)} chart(s)")
    print("=" * 70)
    
    # Display each chart
    for i, url in enumerate(chart_urls, 1):
        chart_name = url.split('/')[-1].replace('.png', '').replace('_', ' ').title()
        print(f"\n{i}. {chart_name}")
        print(f"   {url}")
        
        # Render in notebook
        displayHTML(f'''
        <div style="margin: 25px 0; padding: 20px; 
                    background: linear-gradient(to right, #667eea 0%, #764ba2 100%); 
                    border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <h2 style="color: white; margin: 0 0 15px 0; font-size: 24px;">
                📊 {chart_name}
            </h2>
            <div style="background: white; padding: 15px; border-radius: 8px;">
                <img src="{url}" 
                     style="max-width: 100%; height: auto; display: block; 
                            border-radius: 4px;"/>
            </div>
            <div style="margin-top: 15px; text-align: center;">
                <a href="{url}" target="_blank" 
                   style="color: white; text-decoration: none; 
                          background: rgba(255,255,255,0.25); 
                          padding: 10px 20px; border-radius: 5px; 
                          display: inline-block; font-weight: 600;
                          transition: background 0.3s;">
                    🔗 Open in Full Size
                </a>
            </div>
        </div>
        ''')
    
    print("\n" + "=" * 70)
    print("✅ All charts displayed!")
    
else:
    print("ℹ️  No charts available")
    print("\nCharts are generated when you ask:")
    print("  'What features should I improve to increase my rating?'")
    print("\nMake sure the analysis has completed (~15-20 minutes)")
