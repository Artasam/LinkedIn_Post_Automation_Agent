"""
tools/infographic_engine.py
---------------------------
Dynamic Infographic Engine using Groq structured outputs, 
Mermaid APIs, and HTML-to-Image rendering.
"""

import base64
import logging
import os
import tempfile
from typing import List, Literal, Optional

import requests
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from config import settings

try:
    from html2image import Html2Image
except ImportError:
    Html2Image = None

logger = logging.getLogger(__name__)


# ─── Pydantic Models for Structured Output ────────────────────────────────────

class InfographicType(BaseModel):
    infographic_type: Literal["mermaid_flowchart", "comparison_table", "cheatsheet_list"] = Field(
        description="The best type of infographic for this topic. Use 'mermaid_flowchart' for architectures/processes, 'comparison_table' for comparing two things (VS), and 'cheatsheet_list' for general tips or top N lists."
    )

class ComparisonInfographic(BaseModel):
    title: str = Field(description="The main title of the infographic.")
    subtitle: str = Field(description="A short descriptive subtitle.")
    left_concept: str = Field(description="The first concept being compared.")
    right_concept: str = Field(description="The second concept being compared.")
    left_points: List[str] = Field(description="3 short bullet points for the left concept.", max_length=3)
    right_points: List[str] = Field(description="3 short bullet points for the right concept.", max_length=3)

class CheatSheetInfographic(BaseModel):
    title: str = Field(description="The main title of the cheatsheet.")
    subtitle: str = Field(description="A short descriptive subtitle.")
    points: List[str] = Field(description="5 actionable short bullet points or facts.", max_length=5)

class MermaidInfographic(BaseModel):
    mermaid_code: str = Field(description="Valid mermaid.js syntax for a flowchart. Do NOT include markdown codeblocks (```mermaid), just the raw code. Use 'graph TD'. Keep text inside nodes very short.")


# ─── HTML Templates ───────────────────────────────────────────────────────────

HTML_COMPARISON_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
<style>
body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: linear-gradient(135deg, #0a0e1a 0%, #0d1b2e 100%); color: white; width: 1200px; height: 675px; margin: 0; display: flex; flex-direction: column; align-items: center; justify-content: center; }}
.container {{ width: 90%; background: rgba(255, 255, 255, 0.05); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 20px; padding: 40px; box-shadow: 0 20px 50px rgba(0,0,0,0.5); backdrop-filter: blur(10px); }}
h1 {{ text-align: center; font-size: 48px; margin: 0 0 10px 0; color: #ffffff; text-shadow: 0 0 20px rgba(30, 144, 255, 0.5); }}
h3 {{ text-align: center; font-size: 24px; margin: 0 0 40px 0; color: #4ab8ff; font-weight: 400; }}
.columns {{ display: flex; justify-content: space-between; gap: 40px; }}
.col {{ flex: 1; background: rgba(0, 0, 0, 0.2); padding: 30px; border-radius: 15px; border-top: 4px solid #1e90ff; }}
.col.right {{ border-top: 4px solid #9370db; }}
.col h2 {{ text-align: center; font-size: 36px; margin-top: 0; color: #ffffff; }}
ul {{ list-style-type: none; padding: 0; margin: 0; }}
li {{ font-size: 22px; padding: 15px 0; border-bottom: 1px solid rgba(255,255,255,0.1); line-height: 1.4; }}
li:last-child {{ border-bottom: none; }}
</style>
</head>
<body>
  <div class="container">
    <h1>{title}</h1>
    <h3>{subtitle}</h3>
    <div class="columns">
      <div class="col left">
        <h2>{left_concept}</h2>
        <ul>{left_items}</ul>
      </div>
      <div class="col right">
        <h2>{right_concept}</h2>
        <ul>{right_items}</ul>
      </div>
    </div>
  </div>
</body>
</html>
"""

HTML_CHEATSHEET_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
<style>
body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: linear-gradient(135deg, #0d1b2e 0%, #0a0e1a 100%); color: white; width: 1200px; height: 675px; margin: 0; display: flex; flex-direction: column; align-items: center; justify-content: center; }}
.container {{ width: 85%; background: rgba(255, 255, 255, 0.05); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 20px; padding: 50px; box-shadow: 0 20px 50px rgba(0,0,0,0.5); }}
h1 {{ font-size: 52px; margin: 0 0 10px 0; color: #ffffff; border-left: 8px solid #00bfff; padding-left: 20px; }}
h3 {{ font-size: 26px; margin: 0 0 40px 0; color: #4ab8ff; font-weight: 400; padding-left: 28px; }}
.grid {{ display: grid; grid-template-columns: 1fr; gap: 20px; }}
.card {{ background: rgba(255, 255, 255, 0.03); padding: 25px 30px; border-radius: 12px; font-size: 24px; border-left: 4px solid #1e90ff; display: flex; align-items: center; }}
.card strong {{ color: #00bfff; margin-right: 15px; font-size: 28px; }}
</style>
</head>
<body>
  <div class="container">
    <h1>{title}</h1>
    <h3>{subtitle}</h3>
    <div class="grid">
      {cards}
    </div>
  </div>
</body>
</html>
"""


# ─── Renderers ───────────────────────────────────────────────────────────────

def _render_mermaid(mermaid_code: str) -> Optional[str]:
    """Render Mermaid JS code to PNG via mermaid.ink API."""
    logger.info("Rendering Mermaid diagram via API...")
    try:
        import re
        # Clean up markdown codeblocks if LLM included them
        clean_code = mermaid_code.strip()
        clean_code = re.sub(r"```[a-zA-Z]*", "", clean_code)
        clean_code = clean_code.replace("```", "").strip()
        if clean_code.lower().startswith("mermaid"):
            clean_code = clean_code[7:].strip()
            
        logger.info(f"Cleaned Mermaid Code:\n{clean_code}")
        
        encoded = base64.urlsafe_b64encode(clean_code.encode("utf-8")).decode("utf-8")
        url = f"https://mermaid.ink/img/{encoded}?type=png&bgColor=131b2e"
        
        resp = requests.get(url, timeout=20)
        if resp.status_code == 200:
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False, prefix="linkedin_ai_mermaid_")
            tmp.write(resp.content)
            tmp.close()
            logger.info("Mermaid rendered successfully: %s", tmp.name)
            return tmp.name
        else:
            logger.error("Mermaid API failed: HTTP %d", resp.status_code)
            return None
    except Exception as exc:
        logger.error("Error rendering mermaid: %s", exc)
        return None


def _render_html(html_content: str) -> Optional[str]:
    """Render HTML string to PNG via Html2Image."""
    if Html2Image is None:
        logger.warning("html2image not installed. Run 'pip install html2image'.")
        return None
        
    logger.info("Rendering HTML template to image...")
    try:
        hti = Html2Image(size=(1200, 675))
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False, prefix="linkedin_ai_html_")
        tmp.close() # html2image needs the path, not an open file
        
        # Save HTML to a temporary file
        html_file = tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w", encoding="utf-8")
        html_file.write(html_content)
        html_file.close()
        
        # Render
        out_name = os.path.basename(tmp.name)
        out_dir = os.path.dirname(tmp.name)
        hti.output_path = out_dir
        
        hti.screenshot(html_file=html_file.name, save_as=out_name)
        
        # Cleanup HTML file
        os.remove(html_file.name)
        
        logger.info("HTML rendered successfully: %s", tmp.name)
        return tmp.name
    except Exception as exc:
        logger.error("Error rendering HTML: %s", exc)
        return None


# ─── Core Logic ──────────────────────────────────────────────────────────────

def generate_dynamic_infographic(topic_title: str, topic_summary: str) -> Optional[str]:
    """
    Main entry point: 
    1. Select best infographic type for the topic.
    2. Extract structured data.
    3. Render via HTML or Mermaid.
    """
    if not settings.GROQ_API_KEY:
        logger.warning("GROQ_API_KEY missing for infographic generator.")
        return None

    # Initialize Groq with "openai/gpt-oss-120b" as requested
    llm = ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model="openai/gpt-oss-120b", 
        temperature=0.3,
    )
    
    # Step 1: Select Type
    try:
        logger.info("Infographic Engine: Deciding best format for '%s'...", topic_title)
        router_llm = llm.with_structured_output(InfographicType, method="json_mode")
        decision = router_llm.invoke(
            f"Analyze this topic and choose the best infographic format: {topic_title}. Context: {topic_summary}. "
            "You MUST return a JSON object with exactly this key: "
            "'infographic_type' (string, choose one of: 'mermaid_flowchart', 'comparison_table', 'cheatsheet_list')."
        )
        selected_type = decision.infographic_type
        logger.info("Selected format: %s", selected_type)
    except Exception as exc:
        logger.error("Infographic routing failed: %s", exc)
        # Default fallback
        selected_type = "cheatsheet_list"
        
    # Step 2 & 3: Generate Data and Render
    try:
        if selected_type == "mermaid_flowchart":
            gen_llm = llm.with_structured_output(MermaidInfographic, method="json_mode")
            data = gen_llm.invoke(
                f"Create a mermaid.js flowchart for this AI topic: {topic_title}. Context: {topic_summary}. "
                "Make it visually appealing but simple. You MUST return a JSON object with exactly this key: "
                "'mermaid_code' (string containing the raw mermaid code, no markdown codeblocks)."
            )
            return _render_mermaid(data.mermaid_code)
            
        elif selected_type == "comparison_table":
            gen_llm = llm.with_structured_output(ComparisonInfographic, method="json_mode")
            data = gen_llm.invoke(
                f"Create a comparison infographic for this AI topic: {topic_title}. Context: {topic_summary}. "
                "Find two opposing or complementary concepts to compare. "
                "You MUST return a JSON object with exactly these keys: "
                "'title', 'subtitle', 'left_concept', 'right_concept', 'left_points' (array of 3 strings), 'right_points' (array of 3 strings)."
            )
            
            left_items = "".join(f"<li>{pt}</li>" for pt in data.left_points)
            right_items = "".join(f"<li>{pt}</li>" for pt in data.right_points)
            
            html = HTML_COMPARISON_TEMPLATE.format(
                title=data.title, subtitle=data.subtitle,
                left_concept=data.left_concept, right_concept=data.right_concept,
                left_items=left_items, right_items=right_items
            )
            return _render_html(html)
            
        elif selected_type == "cheatsheet_list":
            gen_llm = llm.with_structured_output(CheatSheetInfographic, method="json_mode")
            data = gen_llm.invoke(
                f"Create a cheat sheet or top facts list for this AI topic: {topic_title}. Context: {topic_summary}. "
                "You MUST return a JSON object with exactly these keys: "
                "'title' (string), 'subtitle' (string), 'points' (array of exactly 5 strings)."
            )
            
            cards = "".join(f"<div class='card'><strong>{i+1}.</strong> {pt}</div>" for i, pt in enumerate(data.points))
            
            html = HTML_CHEATSHEET_TEMPLATE.format(
                title=data.title, subtitle=data.subtitle, cards=cards
            )
            return _render_html(html)
            
    except Exception as exc:
        logger.error("Infographic generation/rendering failed: %s", exc)
        return None
    
    return None
