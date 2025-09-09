from typing import Dict, Any, List
import json
from collections import Counter

IMPACT_KEYS_PRIORITY = [
    "KPMGTotalImpact",
    "PwCTotalImpact",
    "DeloitteTotalImpact",
    "EYTotalImpact",
]

def detect_impact_key(articles: List[Dict[str, Any]]) -> str | None:
    for key in IMPACT_KEYS_PRIORITY:
        if any(isinstance(a.get(key), (int,float)) for a in articles):
            return key
    # Fallback: first key ending with TotalImpact with numeric
    for a in articles:
        for k,v in a.items():
            if k.endswith("TotalImpact") and isinstance(v,(int,float)):
                return k
    return None

def compute_aggregates(articles: List[Dict[str, Any]]) -> Dict[str, Any]:
    impact_key = detect_impact_key(articles)
    impacts = []
    if impact_key:
        impacts = [a.get(impact_key) for a in articles if isinstance(a.get(impact_key),(int,float))]
    total_articles = len(articles)
    avg_impact = sum(impacts)/len(impacts) if impacts else None
    max_row = None
    if impacts:
        max_article = max((a for a in articles if isinstance(a.get(impact_key),(int,float))), key=lambda x: x.get(impact_key,0))
        max_row = {
            "articleid": max_article.get("ArticleID") or None,
            "impact": max_article.get(impact_key),
            "issue": max_article.get("Issue") or None,
            "spokespersonname": max_article.get("SpokespersonName") or None,
        }
    sp_counts = Counter([a.get("SpokespersonName") for a in articles if a.get("SpokespersonName")])
    top_sp = None
    if sp_counts:
        name, cnt = max(sp_counts.items(), key=lambda x: x[1])
        top_sp = {"spokespersonname": name, "article_count": cnt}
    buckets = {"0_1":0, "1_3":0, "3_6":0, "6_10":0}
    for val in impacts:
        if val < 1: buckets["0_1"] +=1
        elif val <3: buckets["1_3"] +=1
        elif val <6: buckets["3_6"] +=1
        else: buckets["6_10"] +=1
    result = {
        "company_impact_field": impact_key,
        "total_articles": total_articles,
        "avg_impact": avg_impact,
        "max_impact": max_row,
        "top_spokesperson": top_sp,
        "impact_distribution": {k:v for k,v in buckets.items() if v>0},
    }
    return result

with open('data/simplified/KPMG_July2024.json', 'r') as f:
    KPMG_July2024 = json.load(f)

with open('data/simplified/PwC_Feb2025.json', 'r') as f:
    PwC_Feb2025 = json.load(f)

with open('data/simplified/EY_Jan2025.json', 'r') as f:
    EY_Jan2025 = json.load(f)

EXAMPLES: List[Dict[str, Any]] = [
    {
        "articles": KPMG_July2024,
        "aggregates": compute_aggregates(KPMG_July2024),
        "summary": (
            "Impact was led by a concentration of birth-rate related coverage spearheaded by Terry Rawnsley. Brendan Rynne provided macro-economic commentary spanning interest rates, productivity and fiscal settings. Minor additional impact came from other spokespeople and syndicated report mentions."
        ),
    },
    {
        "articles": PwC_Feb2025,
        "aggregates": compute_aggregates(PwC_Feb2025),
        "summary": (
            "Minor impact month: a UK Prime Minister reference to a PwC investment ranking dominated sparse coverage; no standout high-impact spokesperson beyond incidental mentions."
        ),
    },
    {
        "articles": EY_Jan2025,
        "aggregates": compute_aggregates(EY_Jan2025), 
        "summary": (
            "Cherelle Murphy (19 Volume) delivered over three-quarters of EY’s economic Impact with commentary on RBA rate speculation following the release of inflation data and potential impact of Trump policies plus the ‘worrying trend’ of low interest in studying economics (ABC-digital/TV/radio, AFR, Sky News). Paula Gadsby (4 V) contributed almost a quarter of Impact with her opinion on RBA action on rates (AFR, The Australian)."
        ),
    }
]

    # { # Deloitte Apr2025
    #     "summary": (
    #         f"Impact led by Pradeep Philip (almost 70% of Deloitte’s Economics Impact), on the housing crisis and affordability, criticising both major political parties for not addressing the root cause being insufficient housing supply; as well as on the Australian economic outlook, uncertainty caused by US tariffs and expected RBA rate cuts. Additional Impact from Adam Vos on likely implications of US tariff policy and on DAE comments regarding cost of living and on the housing agendas of both parties not doing enough to address the crisis."
    #     )
    # },
    # { #KPMG Q1 2025
    #     "summary": (
    #         f"Brendan Rynne (76 clips) delivered 72% of KPMG’s Impact for Economics from his expert commentary. In Jan, Brendan spoke on the housing market, expected RBA rate decision, weakness in job market, retail figures, inflation, potential impact of Trump tariffs, federal and state govt spending and VIC wages; in Feb, provided his expert opinion on the RBA rate cut calling it a ‘lineball call’, pre-election spending promises and impact of US tariffs; and in Mar, on RBA rate prediction, criticism of federal budget and concerns of inflation risk, and Coalition proposal to ease petrol prices. Terry Rawnsley (11 clips) delivered 12% of Economics Impact. In Jan, Terry spoke on KPMG analysis which showed Gen X had more wealth in property and shares than any other generation; in Feb, on the trends in residential property renovations and wage growth; and in Mar, on the fall of donations to charities and up turn in apartment approvals in Sydney and Melbourne. Additional spokespeople included Yael Selfin (spoke on UK inflation, cost of London Heathrow shutdown, global economy and impact of US govt policies), Jai Patel (on the mutual benefit of an economic relationship between Australia and India), Carole Streicher (on implications of US tariff and tax policies), Michael Malakellis (on RBA challenges ahead) and Merriden Varrall (on opportunity for AUS businesses to lead in resilience in cyber around the world and possible US policy effecting global economies). Minor Impact also for mentions of KPMG Budget Calculator and report into housing affordability for young people in Sydney and Melbourne. Minor negative for Andrew Bolt comment on global warming position."
    #     )
    # }