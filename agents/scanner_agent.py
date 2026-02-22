from typing import Optional, List
# from openai import OpenAI   # ❌ Old OpenAI import (kept but commented)
from agents.deals import ScrapedDeal, DealSelection
from agents.agent import Agent

from litellm import completion
import json
import time
from pydantic import ValidationError


class ScannerAgent(Agent):

    # ===============================
    # OLD OPENAI MODEL (COMMENTED)
    # ===============================
    # MODEL = "gpt-5-mini"

    # ===============================
    # NEW GROQ MODEL
    # ===============================
    MODEL = "groq/llama-3.3-70b-versatile"

    SYSTEM_PROMPT = """You identify and summarize the 5 most detailed deals from a list, by selecting deals that have the most detailed, high quality description and the most clear price.
Respond strictly in JSON with no explanation.
You should provide the price as a number derived from the description.
If the price of a deal isn't clear, do not include that deal.
Most important is a thorough product description (not deal terms).
Be careful with "$XXX off" — that is NOT the price.
Only include products when highly confident about price.
"""

    USER_PROMPT_PREFIX = """Respond with the most promising 5 deals from this list.
Select those with the most detailed product description and a clear price > 0.
Rephrase as a summary of the product itself.
Return exactly 5 deals.

Deals:

"""

    USER_PROMPT_SUFFIX = "\n\nInclude exactly 5 deals, no more."

    name = "Scanner Agent"
    color = Agent.CYAN

    def __init__(self):
        self.log("Scanner Agent is initializing (Groq mode)")

        # ===============================
        # OLD OPENAI CLIENT (COMMENTED)
        # ===============================
        # self.openai = OpenAI()

        self.log("Scanner Agent is ready")

    # ---------------------------------------------------
    # FETCH DEALS (UNCHANGED)
    # ---------------------------------------------------

    def fetch_deals(self, memory) -> List[ScrapedDeal]:
        self.log("Scanner Agent is about to fetch deals from RSS feed")
        urls = [opp.deal.url for opp in memory]
        scraped = ScrapedDeal.fetch()
        result = [scrape for scrape in scraped if scrape.url not in urls]
        self.log(f"Scanner Agent received {len(result)} deals not already scraped")
        return result

    # ---------------------------------------------------
    # PROMPT BUILDER (UNCHANGED)
    # ---------------------------------------------------

    def make_user_prompt(self, scraped) -> str:
        user_prompt = self.USER_PROMPT_PREFIX
        user_prompt += "\n\n".join([scrape.describe() for scrape in scraped])
        user_prompt += self.USER_PROMPT_SUFFIX
        return user_prompt

    # ---------------------------------------------------
    # GROQ STRUCTURED PARSE
    # ---------------------------------------------------

    def _groq_parse(self, messages, retries=3):

        format_instruction = {
            "role": "system",
            "content": """
Return ONLY valid JSON in this exact format:

{
  "deals": [
    {
      "product_description": "string",
      "price": number,
      "url": "string"
    }
  ]
}

No markdown.
No explanation.
Raw JSON only.
"""
        }

        for attempt in range(retries):
            response = completion(
                model=self.MODEL,
                messages=[format_instruction] + messages,
                temperature=0,
            )

            content = response.choices[0].message.content.strip()

            # Remove accidental markdown if model adds it
            if content.startswith("```"):
                content = content.replace("```json", "").replace("```", "").strip()

            try:
                data = json.loads(content)
                return DealSelection(**data)
            except (json.JSONDecodeError, ValidationError):
                self.log(f"Retry {attempt+1}: Invalid JSON from Groq")
                time.sleep(1)

        raise ValueError("Groq failed to produce valid structured output.")

    # ---------------------------------------------------
    # SCAN USING GROQ
    # ---------------------------------------------------

    def scan(self, memory: List[str] = []) -> Optional[DealSelection]:

        scraped = self.fetch_deals(memory)

        if not scraped:
            return None

        # 🔥 Optional: limit input size to reduce token usage
        scraped = scraped[:20]

        user_prompt = self.make_user_prompt(scraped)

        self.log("Scanner Agent is calling Groq")

        # ===============================
        # OLD OPENAI STRUCTURED CALL (COMMENTED)
        # ===============================
        """
        result = self.openai.chat.completions.parse(
            model=self.MODEL,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format=DealSelection,
            reasoning_effort="minimal",
        )
        result = result.choices[0].message.parsed
        """

        # ===============================
        # NEW GROQ CALL
        # ===============================

        result = self._groq_parse(
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
        )

        result.deals = [deal for deal in result.deals if deal.price > 0]

        self.log(
            f"Scanner Agent received {len(result.deals)} selected deals with price>0 from Groq"
        )

        return result

    # ---------------------------------------------------
    # TEST METHOD (UNCHANGED)
    # ---------------------------------------------------

    def test_scan(self, memory: List[str] = []) -> Optional[DealSelection]:

        results = {
            "deals": [
                {
                    "product_description": "Example product description",
                    "price": 100,
                    "url": "https://example.com",
                }
            ]
        }

        return DealSelection(**results)