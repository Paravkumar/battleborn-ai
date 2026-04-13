import json
import os
import glob
from pathlib import Path

def generate_report():
    outputs_dir = Path("outputs")
    report_file = outputs_dir / "ticket_summary_report.txt"
    json_files = glob.glob(str(outputs_dir / "workflow_*.json"))
    
    total = len(json_files)
    resolved = 0
    escalated = 0
    failed = 0
    
    ticket_reports = []
    
    for idx, filepath in enumerate(json_files, start=1000):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            continue
            
        status = str(data.get("status", "failed")).upper()
        if status == "COMPLETED" or status == "RESOLVED":
            resolved += 1
        elif status == "ESCALATED":
            escalated += 1
        else:
            failed += 1
            
        customer_msg = "Unknown"
        kb_articles = []
        retries = 0
        model = os.getenv("CLOUD_MODEL", "nvidia/nemotron-4-340b-instruct") # Update to current default
        escalation_reason = ""
        
        for step in data.get("steps", []):
            if step.get("tool") == "ticket_reader" and step.get("outcome") == "success":
                customer_msg = step.get("data", {}).get("customer_message", customer_msg)
            if step.get("tool") == "knowledge_base_query" and step.get("outcome") == "success":
                hits = step.get("data", {}).get("hits", [])
                kb_articles.extend(hits)
            if step.get("attempt", 1) > 1:
                retries += (step.get("attempt", 1) - 1)
            if step.get("tool") == "escalation_trigger":
                escalation_reason = step.get("arguments", {}).get("reason", "")
                
        ticket_id = f"TICK-{idx}"
        
        # Build the string for this ticket
        blocks = []
        blocks.append("======================================================================")
        blocks.append(f"TICKET SUMMARY: {ticket_id}")
        blocks.append("======================================================================")
        
        blocks.append("\n## TICKET INFORMATION")
        blocks.append(f"  Status:     {status}")
        blocks.append(f"  Outcome:    {data.get('final_ticket_status', 'unknown').upper()}")
        
        blocks.append("\n## CUSTOMER REQUEST")
        blocks.append(f"  {customer_msg}")
        
        blocks.append("\n## CONVERSATION HISTORY")
        blocks.append("  [Initial Request]")
        blocks.append(f"    {customer_msg}")
        blocks.append("\n  [Agent Response]")
        blocks.append(f"    {data.get('final_response', '')}")
        
        if kb_articles:
            blocks.append("\n## KNOWLEDGE BASE ARTICLES USED")
            for kb in kb_articles:
                blocks.append(f"  - KB-{kb.get('article_id', 'Unknown')}: {kb.get('title', 'Unknown')}")
                blocks.append(f"    Summary: {kb.get('summary', '')[:80]}...")
                
        if status == "ESCALATED":
            blocks.append("\n## ESCALATION")
            blocks.append(f"  Reason for Escalation:\n    {escalation_reason}")
            
        blocks.append("\n## PROCESSING DETAILS")
        blocks.append(f"  Started:  {data.get('started_at', '')}")
        blocks.append(f"  Finished: {data.get('finished_at', '')}")
        blocks.append(f"  Model:    {model}")
        blocks.append(f"  Retries:  {retries}")
        blocks.append("\n======================================================================\n")
        
        ticket_reports.append("\n".join(blocks))

    # OVERVIEW BLOCK
    overview = []
    overview.append("======================================================================")
    overview.append("TICKET SUMMARY REPORT")
    overview.append("======================================================================")
    overview.append("\n## OVERVIEW")
    overview.append(f"  Total Tickets:     {total}")
    overview.append(f"  Resolved:          {resolved} ({(resolved/max(1, total)*100):.0f}%)")
    overview.append(f"  Escalated:         {escalated} ({(escalated/max(1, total)*100):.0f}%)")
    overview.append(f"  Failed:            {failed}\n")
    overview.append("----------------------------------------------------------------------\n")
    
    final_output = "\n".join(overview) + "".join(ticket_reports)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(final_output, encoding="utf-8")
    print(f"Report generated successfully at {report_file}")

if __name__ == "__main__":
    generate_report()
