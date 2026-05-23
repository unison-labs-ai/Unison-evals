#!/usr/bin/env python3
"""Generate corpus.jsonl and questions.jsonl for BitempoQA v0.

Run once to author the dataset. Output is committed alongside this script.
The script is kept so the generation methodology is transparent.

  python data/bitempoqa/_generate.py
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).parent

# ---------------------------------------------------------------------------
# Corpus — 110 atomic facts, 30 distinct subjects
# Each terminated fact (valid_to != null) has a successor that supersedes it.
# ---------------------------------------------------------------------------

CORPUS: list[dict] = [
    # --- Velox Systems — CEO transitions ---
    {"fact_id": "f001", "subject": "Velox Systems", "predicate": "ceo", "object": "Marta Osei",
     "valid_from": "2021-03-01", "valid_to": "2023-08-15", "supersedes": None, "source_id": "src001"},
    {"fact_id": "f002", "subject": "Velox Systems", "predicate": "ceo", "object": "Derek Holt",
     "valid_from": "2023-08-15", "valid_to": "2025-01-10", "supersedes": "f001", "source_id": "src001"},
    {"fact_id": "f003", "subject": "Velox Systems", "predicate": "ceo", "object": "Priya Nair",
     "valid_from": "2025-01-10", "valid_to": None, "supersedes": "f002", "source_id": "src001"},

    # --- Velox Systems — HQ ---
    {"fact_id": "f004", "subject": "Velox Systems", "predicate": "hq_city", "object": "Austin",
     "valid_from": "2021-03-01", "valid_to": "2024-06-01", "supersedes": None, "source_id": "src002"},
    {"fact_id": "f005", "subject": "Velox Systems", "predicate": "hq_city", "object": "San Francisco",
     "valid_from": "2024-06-01", "valid_to": None, "supersedes": "f004", "source_id": "src002"},

    # --- Orbita Analytics — CEO ---
    {"fact_id": "f006", "subject": "Orbita Analytics", "predicate": "ceo", "object": "Jason Kwon",
     "valid_from": "2020-05-12", "valid_to": "2024-02-28", "supersedes": None, "source_id": "src003"},
    {"fact_id": "f007", "subject": "Orbita Analytics", "predicate": "ceo", "object": "Leila Vasquez",
     "valid_from": "2024-02-28", "valid_to": None, "supersedes": "f006", "source_id": "src003"},

    # --- Orbita Analytics — funding series ---
    {"fact_id": "f008", "subject": "Orbita Analytics", "predicate": "funding_series", "object": "Series A",
     "valid_from": "2020-09-01", "valid_to": "2022-11-15", "supersedes": None, "source_id": "src004"},
    {"fact_id": "f009", "subject": "Orbita Analytics", "predicate": "funding_series", "object": "Series B",
     "valid_from": "2022-11-15", "valid_to": "2024-07-01", "supersedes": "f008", "source_id": "src004"},
    {"fact_id": "f010", "subject": "Orbita Analytics", "predicate": "funding_series", "object": "Series C",
     "valid_from": "2024-07-01", "valid_to": None, "supersedes": "f009", "source_id": "src004"},

    # --- NimbusDrive — CEO ---
    {"fact_id": "f011", "subject": "NimbusDrive", "predicate": "ceo", "object": "Carlos Fuentes",
     "valid_from": "2019-07-20", "valid_to": "2022-03-01", "supersedes": None, "source_id": "src005"},
    {"fact_id": "f012", "subject": "NimbusDrive", "predicate": "ceo", "object": "Aisha Okafor",
     "valid_from": "2022-03-01", "valid_to": None, "supersedes": "f011", "source_id": "src005"},

    # --- NimbusDrive — employee count ---
    {"fact_id": "f013", "subject": "NimbusDrive", "predicate": "employee_count", "object": "45",
     "valid_from": "2019-07-20", "valid_to": "2021-01-01", "supersedes": None, "source_id": "src006"},
    {"fact_id": "f014", "subject": "NimbusDrive", "predicate": "employee_count", "object": "120",
     "valid_from": "2021-01-01", "valid_to": "2023-06-01", "supersedes": "f013", "source_id": "src006"},
    {"fact_id": "f015", "subject": "NimbusDrive", "predicate": "employee_count", "object": "310",
     "valid_from": "2023-06-01", "valid_to": None, "supersedes": "f014", "source_id": "src006"},

    # --- Crestline AI — CEO ---
    {"fact_id": "f016", "subject": "Crestline AI", "predicate": "ceo", "object": "Tobias Meier",
     "valid_from": "2022-01-04", "valid_to": "2024-09-30", "supersedes": None, "source_id": "src007"},
    {"fact_id": "f017", "subject": "Crestline AI", "predicate": "ceo", "object": "Sandra Liu",
     "valid_from": "2024-09-30", "valid_to": None, "supersedes": "f016", "source_id": "src007"},

    # --- Crestline AI — HQ ---
    {"fact_id": "f018", "subject": "Crestline AI", "predicate": "hq_city", "object": "Boston",
     "valid_from": "2022-01-04", "valid_to": None, "supersedes": None, "source_id": "src008"},

    # --- PrismaFlow — product name ---
    {"fact_id": "f019", "subject": "PrismaFlow", "predicate": "flagship_product", "object": "FlowCore v1",
     "valid_from": "2021-06-15", "valid_to": "2023-03-01", "supersedes": None, "source_id": "src009"},
    {"fact_id": "f020", "subject": "PrismaFlow", "predicate": "flagship_product", "object": "FlowCore v2",
     "valid_from": "2023-03-01", "valid_to": "2025-02-01", "supersedes": "f019", "source_id": "src009"},
    {"fact_id": "f021", "subject": "PrismaFlow", "predicate": "flagship_product", "object": "FlowCore v3",
     "valid_from": "2025-02-01", "valid_to": None, "supersedes": "f020", "source_id": "src009"},

    # --- PrismaFlow — CEO ---
    {"fact_id": "f022", "subject": "PrismaFlow", "predicate": "ceo", "object": "Nadia Petrov",
     "valid_from": "2021-06-15", "valid_to": None, "supersedes": None, "source_id": "src010"},

    # --- QuantumLeap Software — CEO ---
    {"fact_id": "f023", "subject": "QuantumLeap Software", "predicate": "ceo", "object": "Henry Walsh",
     "valid_from": "2018-11-01", "valid_to": "2021-07-15", "supersedes": None, "source_id": "src011"},
    {"fact_id": "f024", "subject": "QuantumLeap Software", "predicate": "ceo", "object": "Grace Kim",
     "valid_from": "2021-07-15", "valid_to": "2023-12-01", "supersedes": "f023", "source_id": "src011"},
    {"fact_id": "f025", "subject": "QuantumLeap Software", "predicate": "ceo", "object": "Marcus Reid",
     "valid_from": "2023-12-01", "valid_to": None, "supersedes": "f024", "source_id": "src011"},

    # --- QuantumLeap Software — funding series ---
    {"fact_id": "f026", "subject": "QuantumLeap Software", "predicate": "funding_series", "object": "Seed",
     "valid_from": "2018-11-01", "valid_to": "2020-05-01", "supersedes": None, "source_id": "src012"},
    {"fact_id": "f027", "subject": "QuantumLeap Software", "predicate": "funding_series", "object": "Series A",
     "valid_from": "2020-05-01", "valid_to": None, "supersedes": "f026", "source_id": "src012"},

    # --- Zenith Platforms — CTO ---
    {"fact_id": "f028", "subject": "Zenith Platforms", "predicate": "cto", "object": "Yuki Tanaka",
     "valid_from": "2020-02-01", "valid_to": "2022-10-15", "supersedes": None, "source_id": "src013"},
    {"fact_id": "f029", "subject": "Zenith Platforms", "predicate": "cto", "object": "Oliver Braun",
     "valid_from": "2022-10-15", "valid_to": None, "supersedes": "f028", "source_id": "src013"},

    # --- Zenith Platforms — HQ ---
    {"fact_id": "f030", "subject": "Zenith Platforms", "predicate": "hq_city", "object": "London",
     "valid_from": "2020-02-01", "valid_to": "2023-09-01", "supersedes": None, "source_id": "src014"},
    {"fact_id": "f031", "subject": "Zenith Platforms", "predicate": "hq_city", "object": "Berlin",
     "valid_from": "2023-09-01", "valid_to": None, "supersedes": "f030", "source_id": "src014"},

    # --- Apex Infra — CEO ---
    {"fact_id": "f032", "subject": "Apex Infra", "predicate": "ceo", "object": "Rachel Schwartz",
     "valid_from": "2019-04-10", "valid_to": "2024-01-01", "supersedes": None, "source_id": "src015"},
    {"fact_id": "f033", "subject": "Apex Infra", "predicate": "ceo", "object": "Dimitri Volkov",
     "valid_from": "2024-01-01", "valid_to": None, "supersedes": "f032", "source_id": "src015"},

    # --- Apex Infra — employee count ---
    {"fact_id": "f034", "subject": "Apex Infra", "predicate": "employee_count", "object": "200",
     "valid_from": "2019-04-10", "valid_to": "2022-01-01", "supersedes": None, "source_id": "src016"},
    {"fact_id": "f035", "subject": "Apex Infra", "predicate": "employee_count", "object": "450",
     "valid_from": "2022-01-01", "valid_to": None, "supersedes": "f034", "source_id": "src016"},

    # --- TerraStack — CEO ---
    {"fact_id": "f036", "subject": "TerraStack", "predicate": "ceo", "object": "Ingrid Lund",
     "valid_from": "2021-09-01", "valid_to": "2023-05-20", "supersedes": None, "source_id": "src017"},
    {"fact_id": "f037", "subject": "TerraStack", "predicate": "ceo", "object": "Ben Adeyemi",
     "valid_from": "2023-05-20", "valid_to": None, "supersedes": "f036", "source_id": "src017"},

    # --- TerraStack — partnership ---
    {"fact_id": "f038", "subject": "TerraStack", "predicate": "primary_cloud_partner", "object": "AWS",
     "valid_from": "2021-09-01", "valid_to": "2024-03-01", "supersedes": None, "source_id": "src018"},
    {"fact_id": "f039", "subject": "TerraStack", "predicate": "primary_cloud_partner", "object": "Google Cloud",
     "valid_from": "2024-03-01", "valid_to": None, "supersedes": "f038", "source_id": "src018"},

    # --- Luminary Data — CEO ---
    {"fact_id": "f040", "subject": "Luminary Data", "predicate": "ceo", "object": "Sofia Reyes",
     "valid_from": "2022-07-01", "valid_to": "2025-04-15", "supersedes": None, "source_id": "src019"},
    {"fact_id": "f041", "subject": "Luminary Data", "predicate": "ceo", "object": "Tyler Grant",
     "valid_from": "2025-04-15", "valid_to": None, "supersedes": "f040", "source_id": "src019"},

    # --- Luminary Data — HQ ---
    {"fact_id": "f042", "subject": "Luminary Data", "predicate": "hq_city", "object": "Chicago",
     "valid_from": "2022-07-01", "valid_to": None, "supersedes": None, "source_id": "src020"},

    # --- Cascade Networks — CEO ---
    {"fact_id": "f043", "subject": "Cascade Networks", "predicate": "ceo", "object": "Wen Zhang",
     "valid_from": "2017-03-15", "valid_to": "2020-08-01", "supersedes": None, "source_id": "src021"},
    {"fact_id": "f044", "subject": "Cascade Networks", "predicate": "ceo", "object": "Fatima Al-Hassan",
     "valid_from": "2020-08-01", "valid_to": "2023-11-30", "supersedes": "f043", "source_id": "src021"},
    {"fact_id": "f045", "subject": "Cascade Networks", "predicate": "ceo", "object": "Marcus Bell",
     "valid_from": "2023-11-30", "valid_to": None, "supersedes": "f044", "source_id": "src021"},

    # --- Cascade Networks — funding series ---
    {"fact_id": "f046", "subject": "Cascade Networks", "predicate": "funding_series", "object": "Series B",
     "valid_from": "2019-06-01", "valid_to": "2022-02-01", "supersedes": None, "source_id": "src022"},
    {"fact_id": "f047", "subject": "Cascade Networks", "predicate": "funding_series", "object": "Series C",
     "valid_from": "2022-02-01", "valid_to": None, "supersedes": "f046", "source_id": "src022"},

    # --- Ironclad Systems — CTO ---
    {"fact_id": "f048", "subject": "Ironclad Systems", "predicate": "cto", "object": "Ravi Sharma",
     "valid_from": "2020-11-01", "valid_to": "2024-04-01", "supersedes": None, "source_id": "src023"},
    {"fact_id": "f049", "subject": "Ironclad Systems", "predicate": "cto", "object": "Elena Torres",
     "valid_from": "2024-04-01", "valid_to": None, "supersedes": "f048", "source_id": "src023"},

    # --- Ironclad Systems — HQ ---
    {"fact_id": "f050", "subject": "Ironclad Systems", "predicate": "hq_city", "object": "Seattle",
     "valid_from": "2020-11-01", "valid_to": "2023-01-15", "supersedes": None, "source_id": "src024"},
    {"fact_id": "f051", "subject": "Ironclad Systems", "predicate": "hq_city", "object": "Denver",
     "valid_from": "2023-01-15", "valid_to": None, "supersedes": "f050", "source_id": "src024"},

    # --- SkyBridge Tech — CEO ---
    {"fact_id": "f052", "subject": "SkyBridge Tech", "predicate": "ceo", "object": "Alicia Monroe",
     "valid_from": "2023-01-01", "valid_to": "2025-06-01", "supersedes": None, "source_id": "src025"},
    {"fact_id": "f053", "subject": "SkyBridge Tech", "predicate": "ceo", "object": "Nathan Cho",
     "valid_from": "2025-06-01", "valid_to": None, "supersedes": "f052", "source_id": "src025"},

    # --- SkyBridge Tech — flagship product ---
    {"fact_id": "f054", "subject": "SkyBridge Tech", "predicate": "flagship_product", "object": "BridgeOS 1.0",
     "valid_from": "2023-01-01", "valid_to": "2024-09-01", "supersedes": None, "source_id": "src026"},
    {"fact_id": "f055", "subject": "SkyBridge Tech", "predicate": "flagship_product", "object": "BridgeOS 2.0",
     "valid_from": "2024-09-01", "valid_to": None, "supersedes": "f054", "source_id": "src026"},

    # --- Pegasus Cloud — CEO ---
    {"fact_id": "f056", "subject": "Pegasus Cloud", "predicate": "ceo", "object": "Diana Osei",
     "valid_from": "2016-10-01", "valid_to": "2021-03-15", "supersedes": None, "source_id": "src027"},
    {"fact_id": "f057", "subject": "Pegasus Cloud", "predicate": "ceo", "object": "James Nakamura",
     "valid_from": "2021-03-15", "valid_to": None, "supersedes": "f056", "source_id": "src027"},

    # --- Pegasus Cloud — employee count ---
    {"fact_id": "f058", "subject": "Pegasus Cloud", "predicate": "employee_count", "object": "80",
     "valid_from": "2016-10-01", "valid_to": "2020-01-01", "supersedes": None, "source_id": "src028"},
    {"fact_id": "f059", "subject": "Pegasus Cloud", "predicate": "employee_count", "object": "225",
     "valid_from": "2020-01-01", "valid_to": "2023-07-01", "supersedes": "f058", "source_id": "src028"},
    {"fact_id": "f060", "subject": "Pegasus Cloud", "predicate": "employee_count", "object": "580",
     "valid_from": "2023-07-01", "valid_to": None, "supersedes": "f059", "source_id": "src028"},

    # --- Solaris DevTools — CEO ---
    {"fact_id": "f061", "subject": "Solaris DevTools", "predicate": "ceo", "object": "Kevin Park",
     "valid_from": "2020-04-01", "valid_to": "2022-12-01", "supersedes": None, "source_id": "src029"},
    {"fact_id": "f062", "subject": "Solaris DevTools", "predicate": "ceo", "object": "Amara Diallo",
     "valid_from": "2022-12-01", "valid_to": None, "supersedes": "f061", "source_id": "src029"},

    # --- Solaris DevTools — HQ ---
    {"fact_id": "f063", "subject": "Solaris DevTools", "predicate": "hq_city", "object": "Portland",
     "valid_from": "2020-04-01", "valid_to": "2024-07-01", "supersedes": None, "source_id": "src030"},
    {"fact_id": "f064", "subject": "Solaris DevTools", "predicate": "hq_city", "object": "Austin",
     "valid_from": "2024-07-01", "valid_to": None, "supersedes": "f063", "source_id": "src030"},

    # --- Aurora Robotics — CEO ---
    {"fact_id": "f065", "subject": "Aurora Robotics", "predicate": "ceo", "object": "Sven Eriksson",
     "valid_from": "2021-02-28", "valid_to": "2023-10-01", "supersedes": None, "source_id": "src031"},
    {"fact_id": "f066", "subject": "Aurora Robotics", "predicate": "ceo", "object": "Priyanka Menon",
     "valid_from": "2023-10-01", "valid_to": None, "supersedes": "f065", "source_id": "src031"},

    # --- Aurora Robotics — funding series ---
    {"fact_id": "f067", "subject": "Aurora Robotics", "predicate": "funding_series", "object": "Series A",
     "valid_from": "2021-08-01", "valid_to": "2024-01-15", "supersedes": None, "source_id": "src032"},
    {"fact_id": "f068", "subject": "Aurora Robotics", "predicate": "funding_series", "object": "Series B",
     "valid_from": "2024-01-15", "valid_to": None, "supersedes": "f067", "source_id": "src032"},

    # --- Meridian Labs — CEO ---
    {"fact_id": "f069", "subject": "Meridian Labs", "predicate": "ceo", "object": "Chloe Beaumont",
     "valid_from": "2019-11-01", "valid_to": "2022-05-15", "supersedes": None, "source_id": "src033"},
    {"fact_id": "f070", "subject": "Meridian Labs", "predicate": "ceo", "object": "Adrian Cross",
     "valid_from": "2022-05-15", "valid_to": None, "supersedes": "f069", "source_id": "src033"},

    # --- Meridian Labs — primary_cloud_partner ---
    {"fact_id": "f071", "subject": "Meridian Labs", "predicate": "primary_cloud_partner", "object": "Azure",
     "valid_from": "2019-11-01", "valid_to": "2023-08-01", "supersedes": None, "source_id": "src034"},
    {"fact_id": "f072", "subject": "Meridian Labs", "predicate": "primary_cloud_partner", "object": "AWS",
     "valid_from": "2023-08-01", "valid_to": None, "supersedes": "f071", "source_id": "src034"},

    # --- CoralByte — CEO ---
    {"fact_id": "f073", "subject": "CoralByte", "predicate": "ceo", "object": "Liang Xu",
     "valid_from": "2022-03-10", "valid_to": "2024-11-01", "supersedes": None, "source_id": "src035"},
    {"fact_id": "f074", "subject": "CoralByte", "predicate": "ceo", "object": "Hannah White",
     "valid_from": "2024-11-01", "valid_to": None, "supersedes": "f073", "source_id": "src035"},

    # --- CoralByte — HQ ---
    {"fact_id": "f075", "subject": "CoralByte", "predicate": "hq_city", "object": "Miami",
     "valid_from": "2022-03-10", "valid_to": None, "supersedes": None, "source_id": "src036"},

    # --- Helix Technologies — CTO ---
    {"fact_id": "f076", "subject": "Helix Technologies", "predicate": "cto", "object": "Morris Chen",
     "valid_from": "2018-07-01", "valid_to": "2021-04-01", "supersedes": None, "source_id": "src037"},
    {"fact_id": "f077", "subject": "Helix Technologies", "predicate": "cto", "object": "Isabella Rossi",
     "valid_from": "2021-04-01", "valid_to": "2024-06-15", "supersedes": "f076", "source_id": "src037"},
    {"fact_id": "f078", "subject": "Helix Technologies", "predicate": "cto", "object": "Arthur Flynn",
     "valid_from": "2024-06-15", "valid_to": None, "supersedes": "f077", "source_id": "src037"},

    # --- Helix Technologies — employee count ---
    {"fact_id": "f079", "subject": "Helix Technologies", "predicate": "employee_count", "object": "30",
     "valid_from": "2018-07-01", "valid_to": "2021-01-01", "supersedes": None, "source_id": "src038"},
    {"fact_id": "f080", "subject": "Helix Technologies", "predicate": "employee_count", "object": "150",
     "valid_from": "2021-01-01", "valid_to": None, "supersedes": "f079", "source_id": "src038"},

    # --- Trident SaaS — CEO ---
    {"fact_id": "f081", "subject": "Trident SaaS", "predicate": "ceo", "object": "Nora Fleming",
     "valid_from": "2020-08-01", "valid_to": "2023-02-28", "supersedes": None, "source_id": "src039"},
    {"fact_id": "f082", "subject": "Trident SaaS", "predicate": "ceo", "object": "Cyrus Ahmadi",
     "valid_from": "2023-02-28", "valid_to": None, "supersedes": "f081", "source_id": "src039"},

    # --- Trident SaaS — funding series ---
    {"fact_id": "f083", "subject": "Trident SaaS", "predicate": "funding_series", "object": "Seed",
     "valid_from": "2020-08-01", "valid_to": "2022-04-01", "supersedes": None, "source_id": "src040"},
    {"fact_id": "f084", "subject": "Trident SaaS", "predicate": "funding_series", "object": "Series A",
     "valid_from": "2022-04-01", "valid_to": "2024-10-01", "supersedes": "f083", "source_id": "src040"},
    {"fact_id": "f085", "subject": "Trident SaaS", "predicate": "funding_series", "object": "Series B",
     "valid_from": "2024-10-01", "valid_to": None, "supersedes": "f084", "source_id": "src040"},

    # --- Nimbus Edge — CEO ---
    {"fact_id": "f086", "subject": "Nimbus Edge", "predicate": "ceo", "object": "Tomoko Hayashi",
     "valid_from": "2023-04-01", "valid_to": None, "supersedes": None, "source_id": "src041"},

    # --- Nimbus Edge — HQ ---
    {"fact_id": "f087", "subject": "Nimbus Edge", "predicate": "hq_city", "object": "Tokyo",
     "valid_from": "2023-04-01", "valid_to": "2025-01-01", "supersedes": None, "source_id": "src042"},
    {"fact_id": "f088", "subject": "Nimbus Edge", "predicate": "hq_city", "object": "Singapore",
     "valid_from": "2025-01-01", "valid_to": None, "supersedes": "f087", "source_id": "src042"},

    # --- Cobalt Compute — CEO ---
    {"fact_id": "f089", "subject": "Cobalt Compute", "predicate": "ceo", "object": "Ray Johansson",
     "valid_from": "2017-06-01", "valid_to": "2020-10-01", "supersedes": None, "source_id": "src043"},
    {"fact_id": "f090", "subject": "Cobalt Compute", "predicate": "ceo", "object": "Maya Gupta",
     "valid_from": "2020-10-01", "valid_to": None, "supersedes": "f089", "source_id": "src043"},

    # --- Cobalt Compute — primary_cloud_partner ---
    {"fact_id": "f091", "subject": "Cobalt Compute", "predicate": "primary_cloud_partner", "object": "GCP",
     "valid_from": "2017-06-01", "valid_to": "2022-06-01", "supersedes": None, "source_id": "src044"},
    {"fact_id": "f092", "subject": "Cobalt Compute", "predicate": "primary_cloud_partner", "object": "AWS",
     "valid_from": "2022-06-01", "valid_to": None, "supersedes": "f091", "source_id": "src044"},

    # --- VertexOps — CEO ---
    {"fact_id": "f093", "subject": "VertexOps", "predicate": "ceo", "object": "Sarah McAllister",
     "valid_from": "2021-05-01", "valid_to": "2024-03-01", "supersedes": None, "source_id": "src045"},
    {"fact_id": "f094", "subject": "VertexOps", "predicate": "ceo", "object": "Leo Brennan",
     "valid_from": "2024-03-01", "valid_to": None, "supersedes": "f093", "source_id": "src045"},

    # --- VertexOps — flagship_product ---
    {"fact_id": "f095", "subject": "VertexOps", "predicate": "flagship_product", "object": "OpsHub 1.0",
     "valid_from": "2021-05-01", "valid_to": "2023-07-15", "supersedes": None, "source_id": "src046"},
    {"fact_id": "f096", "subject": "VertexOps", "predicate": "flagship_product", "object": "OpsHub 2.0",
     "valid_from": "2023-07-15", "valid_to": None, "supersedes": "f095", "source_id": "src046"},

    # --- PineCore — CEO ---
    {"fact_id": "f097", "subject": "PineCore", "predicate": "ceo", "object": "Thomas Berger",
     "valid_from": "2019-08-01", "valid_to": "2022-09-01", "supersedes": None, "source_id": "src047"},
    {"fact_id": "f098", "subject": "PineCore", "predicate": "ceo", "object": "Anaya Patel",
     "valid_from": "2022-09-01", "valid_to": None, "supersedes": "f097", "source_id": "src047"},

    # --- PineCore — HQ ---
    {"fact_id": "f099", "subject": "PineCore", "predicate": "hq_city", "object": "Toronto",
     "valid_from": "2019-08-01", "valid_to": "2023-04-15", "supersedes": None, "source_id": "src048"},
    {"fact_id": "f100", "subject": "PineCore", "predicate": "hq_city", "object": "New York",
     "valid_from": "2023-04-15", "valid_to": None, "supersedes": "f099", "source_id": "src048"},

    # --- Dawnlight AI — CEO ---
    {"fact_id": "f101", "subject": "Dawnlight AI", "predicate": "ceo", "object": "Victor Osei",
     "valid_from": "2020-01-15", "valid_to": "2022-07-01", "supersedes": None, "source_id": "src049"},
    {"fact_id": "f102", "subject": "Dawnlight AI", "predicate": "ceo", "object": "Elena Novak",
     "valid_from": "2022-07-01", "valid_to": None, "supersedes": "f101", "source_id": "src049"},

    # --- Dawnlight AI — funding series ---
    {"fact_id": "f103", "subject": "Dawnlight AI", "predicate": "funding_series", "object": "Seed",
     "valid_from": "2020-01-15", "valid_to": "2021-09-01", "supersedes": None, "source_id": "src050"},
    {"fact_id": "f104", "subject": "Dawnlight AI", "predicate": "funding_series", "object": "Series A",
     "valid_from": "2021-09-01", "valid_to": None, "supersedes": "f103", "source_id": "src050"},

    # --- StormPath Analytics — CEO ---
    {"fact_id": "f105", "subject": "StormPath Analytics", "predicate": "ceo", "object": "Kenji Watanabe",
     "valid_from": "2022-06-01", "valid_to": "2025-02-28", "supersedes": None, "source_id": "src051"},
    {"fact_id": "f106", "subject": "StormPath Analytics", "predicate": "ceo", "object": "Rosa Martinez",
     "valid_from": "2025-02-28", "valid_to": None, "supersedes": "f105", "source_id": "src051"},

    # --- StormPath Analytics — HQ ---
    {"fact_id": "f107", "subject": "StormPath Analytics", "predicate": "hq_city", "object": "Madrid",
     "valid_from": "2022-06-01", "valid_to": None, "supersedes": None, "source_id": "src052"},

    # --- BrightGrid Solutions — CTO ---
    {"fact_id": "f108", "subject": "BrightGrid Solutions", "predicate": "cto", "object": "Paul Okonkwo",
     "valid_from": "2021-10-01", "valid_to": "2024-05-15", "supersedes": None, "source_id": "src053"},
    {"fact_id": "f109", "subject": "BrightGrid Solutions", "predicate": "cto", "object": "Ji-Yeon Park",
     "valid_from": "2024-05-15", "valid_to": None, "supersedes": "f108", "source_id": "src053"},

    # --- BrightGrid Solutions — employee count ---
    {"fact_id": "f110", "subject": "BrightGrid Solutions", "predicate": "employee_count", "object": "60",
     "valid_from": "2021-10-01", "valid_to": None, "supersedes": None, "source_id": "src054"},
]


# ---------------------------------------------------------------------------
# Questions — 100 questions, ~25 per type, difficulty 1-3
# ---------------------------------------------------------------------------

QUESTIONS: list[dict] = [
    # ===== current_truth (25) =====
    {"id": "q001", "question": "Who is Velox Systems' current CEO?",
     "expected_answer": "Priya Nair", "as_of": None,
     "fact_ids": ["f003"], "question_type": "current_truth", "difficulty": 1},

    {"id": "q002", "question": "What city is Velox Systems' headquarters currently in?",
     "expected_answer": "San Francisco", "as_of": None,
     "fact_ids": ["f005"], "question_type": "current_truth", "difficulty": 1},

    {"id": "q003", "question": "Who is the current CEO of Orbita Analytics?",
     "expected_answer": "Leila Vasquez", "as_of": None,
     "fact_ids": ["f007"], "question_type": "current_truth", "difficulty": 1},

    {"id": "q004", "question": "What is Orbita Analytics' current funding series?",
     "expected_answer": "Series C", "as_of": None,
     "fact_ids": ["f010"], "question_type": "current_truth", "difficulty": 1},

    {"id": "q005", "question": "Who is the current CEO of NimbusDrive?",
     "expected_answer": "Aisha Okafor", "as_of": None,
     "fact_ids": ["f012"], "question_type": "current_truth", "difficulty": 1},

    {"id": "q006", "question": "How many employees does NimbusDrive currently have?",
     "expected_answer": "310", "as_of": None,
     "fact_ids": ["f015"], "question_type": "current_truth", "difficulty": 1},

    {"id": "q007", "question": "Who is Crestline AI's current CEO?",
     "expected_answer": "Sandra Liu", "as_of": None,
     "fact_ids": ["f017"], "question_type": "current_truth", "difficulty": 1},

    {"id": "q008", "question": "What is PrismaFlow's current flagship product?",
     "expected_answer": "FlowCore v3", "as_of": None,
     "fact_ids": ["f021"], "question_type": "current_truth", "difficulty": 1},

    {"id": "q009", "question": "Who is the current CEO of QuantumLeap Software?",
     "expected_answer": "Marcus Reid", "as_of": None,
     "fact_ids": ["f025"], "question_type": "current_truth", "difficulty": 1},

    {"id": "q010", "question": "What city is Zenith Platforms' headquarters currently in?",
     "expected_answer": "Berlin", "as_of": None,
     "fact_ids": ["f031"], "question_type": "current_truth", "difficulty": 1},

    {"id": "q011", "question": "Who is the current CEO of Apex Infra?",
     "expected_answer": "Dimitri Volkov", "as_of": None,
     "fact_ids": ["f033"], "question_type": "current_truth", "difficulty": 1},

    {"id": "q012", "question": "Who is the current CEO of TerraStack?",
     "expected_answer": "Ben Adeyemi", "as_of": None,
     "fact_ids": ["f037"], "question_type": "current_truth", "difficulty": 1},

    {"id": "q013", "question": "What is TerraStack's current primary cloud partner?",
     "expected_answer": "Google Cloud", "as_of": None,
     "fact_ids": ["f039"], "question_type": "current_truth", "difficulty": 1},

    {"id": "q014", "question": "Who is Luminary Data's current CEO?",
     "expected_answer": "Tyler Grant", "as_of": None,
     "fact_ids": ["f041"], "question_type": "current_truth", "difficulty": 1},

    {"id": "q015", "question": "Who is the current CEO of Cascade Networks?",
     "expected_answer": "Marcus Bell", "as_of": None,
     "fact_ids": ["f045"], "question_type": "current_truth", "difficulty": 1},

    {"id": "q016", "question": "Who is the current CTO of Ironclad Systems?",
     "expected_answer": "Elena Torres", "as_of": None,
     "fact_ids": ["f049"], "question_type": "current_truth", "difficulty": 1},

    {"id": "q017", "question": "Who is SkyBridge Tech's current CEO?",
     "expected_answer": "Nathan Cho", "as_of": None,
     "fact_ids": ["f053"], "question_type": "current_truth", "difficulty": 1},

    {"id": "q018", "question": "What is SkyBridge Tech's current flagship product?",
     "expected_answer": "BridgeOS 2.0", "as_of": None,
     "fact_ids": ["f055"], "question_type": "current_truth", "difficulty": 1},

    {"id": "q019", "question": "Who is Solaris DevTools' current CEO?",
     "expected_answer": "Amara Diallo", "as_of": None,
     "fact_ids": ["f062"], "question_type": "current_truth", "difficulty": 1},

    {"id": "q020", "question": "Who is the current CTO of Helix Technologies?",
     "expected_answer": "Arthur Flynn", "as_of": None,
     "fact_ids": ["f078"], "question_type": "current_truth", "difficulty": 1},

    {"id": "q021", "question": "Who is the current CEO of Trident SaaS?",
     "expected_answer": "Cyrus Ahmadi", "as_of": None,
     "fact_ids": ["f082"], "question_type": "current_truth", "difficulty": 1},

    {"id": "q022", "question": "What is Trident SaaS' current funding series?",
     "expected_answer": "Series B", "as_of": None,
     "fact_ids": ["f085"], "question_type": "current_truth", "difficulty": 1},

    {"id": "q023", "question": "Who is VertexOps' current CEO?",
     "expected_answer": "Leo Brennan", "as_of": None,
     "fact_ids": ["f094"], "question_type": "current_truth", "difficulty": 1},

    {"id": "q024", "question": "Who is the current CEO of StormPath Analytics?",
     "expected_answer": "Rosa Martinez", "as_of": None,
     "fact_ids": ["f106"], "question_type": "current_truth", "difficulty": 1},

    {"id": "q025", "question": "Who is the current CTO of BrightGrid Solutions?",
     "expected_answer": "Ji-Yeon Park", "as_of": None,
     "fact_ids": ["f109"], "question_type": "current_truth", "difficulty": 1},

    # ===== historical_truth (25) =====
    {"id": "q026", "question": "Who was Velox Systems' CEO on 2022-06-01?",
     "expected_answer": "Marta Osei", "as_of": "2022-06-01",
     "fact_ids": ["f001"], "question_type": "historical_truth", "difficulty": 2},

    {"id": "q027", "question": "What city was Velox Systems headquartered in on 2023-01-01?",
     "expected_answer": "Austin", "as_of": "2023-01-01",
     "fact_ids": ["f004"], "question_type": "historical_truth", "difficulty": 2},

    {"id": "q028", "question": "Who was the CEO of Orbita Analytics on 2023-06-15?",
     "expected_answer": "Jason Kwon", "as_of": "2023-06-15",
     "fact_ids": ["f006"], "question_type": "historical_truth", "difficulty": 2},

    {"id": "q029", "question": "What funding series was Orbita Analytics on at the start of 2023?",
     "expected_answer": "Series B", "as_of": "2023-01-01",
     "fact_ids": ["f009"], "question_type": "historical_truth", "difficulty": 2},

    {"id": "q030", "question": "Who was NimbusDrive's CEO on 2021-06-01?",
     "expected_answer": "Carlos Fuentes", "as_of": "2021-06-01",
     "fact_ids": ["f011"], "question_type": "historical_truth", "difficulty": 2},

    {"id": "q031", "question": "How many employees did NimbusDrive have on 2022-01-01?",
     "expected_answer": "120", "as_of": "2022-01-01",
     "fact_ids": ["f014"], "question_type": "historical_truth", "difficulty": 2},

    {"id": "q032", "question": "Who was Crestline AI's CEO on 2024-01-15?",
     "expected_answer": "Tobias Meier", "as_of": "2024-01-15",
     "fact_ids": ["f016"], "question_type": "historical_truth", "difficulty": 2},

    {"id": "q033", "question": "What was PrismaFlow's flagship product on 2023-06-01?",
     "expected_answer": "FlowCore v2", "as_of": "2023-06-01",
     "fact_ids": ["f020"], "question_type": "historical_truth", "difficulty": 2},

    {"id": "q034", "question": "Who was QuantumLeap Software's CEO on 2022-03-01?",
     "expected_answer": "Grace Kim", "as_of": "2022-03-01",
     "fact_ids": ["f024"], "question_type": "historical_truth", "difficulty": 2},

    {"id": "q035", "question": "What city was Zenith Platforms headquartered in on 2022-12-01?",
     "expected_answer": "London", "as_of": "2022-12-01",
     "fact_ids": ["f030"], "question_type": "historical_truth", "difficulty": 2},

    {"id": "q036", "question": "Who was Apex Infra's CEO on 2023-05-01?",
     "expected_answer": "Rachel Schwartz", "as_of": "2023-05-01",
     "fact_ids": ["f032"], "question_type": "historical_truth", "difficulty": 2},

    {"id": "q037", "question": "What was TerraStack's primary cloud partner on 2023-01-01?",
     "expected_answer": "AWS", "as_of": "2023-01-01",
     "fact_ids": ["f038"], "question_type": "historical_truth", "difficulty": 2},

    {"id": "q038", "question": "Who was the CEO of Cascade Networks on 2021-01-01?",
     "expected_answer": "Fatima Al-Hassan", "as_of": "2021-01-01",
     "fact_ids": ["f044"], "question_type": "historical_truth", "difficulty": 2},

    {"id": "q039", "question": "What city was Ironclad Systems headquartered in on 2022-06-01?",
     "expected_answer": "Seattle", "as_of": "2022-06-01",
     "fact_ids": ["f050"], "question_type": "historical_truth", "difficulty": 2},

    {"id": "q040", "question": "Who was Pegasus Cloud's CEO on 2020-01-01?",
     "expected_answer": "Diana Osei", "as_of": "2020-01-01",
     "fact_ids": ["f056"], "question_type": "historical_truth", "difficulty": 2},

    {"id": "q041", "question": "How many employees did Pegasus Cloud have on 2021-06-01?",
     "expected_answer": "225", "as_of": "2021-06-01",
     "fact_ids": ["f059"], "question_type": "historical_truth", "difficulty": 2},

    {"id": "q042", "question": "Who was the CTO of Helix Technologies on 2022-01-01?",
     "expected_answer": "Isabella Rossi", "as_of": "2022-01-01",
     "fact_ids": ["f077"], "question_type": "historical_truth", "difficulty": 2},

    {"id": "q043", "question": "What was Meridian Labs' primary cloud partner on 2022-05-01?",
     "expected_answer": "Azure", "as_of": "2022-05-01",
     "fact_ids": ["f071"], "question_type": "historical_truth", "difficulty": 2},

    {"id": "q044", "question": "Who was CoralByte's CEO on 2023-08-01?",
     "expected_answer": "Liang Xu", "as_of": "2023-08-01",
     "fact_ids": ["f073"], "question_type": "historical_truth", "difficulty": 2},

    {"id": "q045", "question": "What city was Nimbus Edge headquartered in on 2024-06-01?",
     "expected_answer": "Tokyo", "as_of": "2024-06-01",
     "fact_ids": ["f087"], "question_type": "historical_truth", "difficulty": 2},

    {"id": "q046", "question": "Who was Cobalt Compute's CEO on 2019-01-01?",
     "expected_answer": "Ray Johansson", "as_of": "2019-01-01",
     "fact_ids": ["f089"], "question_type": "historical_truth", "difficulty": 2},

    {"id": "q047", "question": "Who was Trident SaaS' CEO on 2022-01-01?",
     "expected_answer": "Nora Fleming", "as_of": "2022-01-01",
     "fact_ids": ["f081"], "question_type": "historical_truth", "difficulty": 2},

    {"id": "q048", "question": "What funding series was Dawnlight AI on in early 2021?",
     "expected_answer": "Seed", "as_of": "2021-03-01",
     "fact_ids": ["f103"], "question_type": "historical_truth", "difficulty": 2},

    {"id": "q049", "question": "Who was Velox Systems' CEO on 2024-01-01?",
     "expected_answer": "Derek Holt", "as_of": "2024-01-01",
     "fact_ids": ["f002"], "question_type": "historical_truth", "difficulty": 2},

    {"id": "q050", "question": "What city was Solaris DevTools headquartered in on 2023-01-01?",
     "expected_answer": "Portland", "as_of": "2023-01-01",
     "fact_ids": ["f063"], "question_type": "historical_truth", "difficulty": 2},

    # ===== predecessor (25) =====
    {"id": "q051", "question": "Who was the CEO of Velox Systems before Priya Nair?",
     "expected_answer": "Derek Holt", "as_of": None,
     "fact_ids": ["f002", "f003"], "question_type": "predecessor", "difficulty": 3},

    {"id": "q052", "question": "Who was the CEO of Velox Systems before Derek Holt?",
     "expected_answer": "Marta Osei", "as_of": None,
     "fact_ids": ["f001", "f002"], "question_type": "predecessor", "difficulty": 3},

    {"id": "q053", "question": "Who was the CEO of Orbita Analytics before Leila Vasquez?",
     "expected_answer": "Jason Kwon", "as_of": None,
     "fact_ids": ["f006", "f007"], "question_type": "predecessor", "difficulty": 3},

    {"id": "q054", "question": "Who led NimbusDrive before Aisha Okafor became CEO?",
     "expected_answer": "Carlos Fuentes", "as_of": None,
     "fact_ids": ["f011", "f012"], "question_type": "predecessor", "difficulty": 3},

    {"id": "q055", "question": "Who was the CEO of Crestline AI before Sandra Liu?",
     "expected_answer": "Tobias Meier", "as_of": None,
     "fact_ids": ["f016", "f017"], "question_type": "predecessor", "difficulty": 3},

    {"id": "q056", "question": "What was PrismaFlow's flagship product before FlowCore v3?",
     "expected_answer": "FlowCore v2", "as_of": None,
     "fact_ids": ["f020", "f021"], "question_type": "predecessor", "difficulty": 3},

    {"id": "q057", "question": "What was PrismaFlow's flagship product before FlowCore v2?",
     "expected_answer": "FlowCore v1", "as_of": None,
     "fact_ids": ["f019", "f020"], "question_type": "predecessor", "difficulty": 3},

    {"id": "q058", "question": "Who was QuantumLeap Software's CEO before Marcus Reid?",
     "expected_answer": "Grace Kim", "as_of": None,
     "fact_ids": ["f024", "f025"], "question_type": "predecessor", "difficulty": 3},

    {"id": "q059", "question": "Who was QuantumLeap Software's CEO before Grace Kim?",
     "expected_answer": "Henry Walsh", "as_of": None,
     "fact_ids": ["f023", "f024"], "question_type": "predecessor", "difficulty": 3},

    {"id": "q060", "question": "What city was Zenith Platforms headquartered in before Berlin?",
     "expected_answer": "London", "as_of": None,
     "fact_ids": ["f030", "f031"], "question_type": "predecessor", "difficulty": 3},

    {"id": "q061", "question": "Who was the CEO of Apex Infra before Dimitri Volkov?",
     "expected_answer": "Rachel Schwartz", "as_of": None,
     "fact_ids": ["f032", "f033"], "question_type": "predecessor", "difficulty": 3},

    {"id": "q062", "question": "Who was TerraStack's CEO before Ben Adeyemi?",
     "expected_answer": "Ingrid Lund", "as_of": None,
     "fact_ids": ["f036", "f037"], "question_type": "predecessor", "difficulty": 3},

    {"id": "q063", "question": "What was TerraStack's primary cloud partner before Google Cloud?",
     "expected_answer": "AWS", "as_of": None,
     "fact_ids": ["f038", "f039"], "question_type": "predecessor", "difficulty": 3},

    {"id": "q064", "question": "Who was the CEO of Cascade Networks before Marcus Bell?",
     "expected_answer": "Fatima Al-Hassan", "as_of": None,
     "fact_ids": ["f044", "f045"], "question_type": "predecessor", "difficulty": 3},

    {"id": "q065", "question": "Who was the CEO of Cascade Networks before Fatima Al-Hassan?",
     "expected_answer": "Wen Zhang", "as_of": None,
     "fact_ids": ["f043", "f044"], "question_type": "predecessor", "difficulty": 3},

    {"id": "q066", "question": "What city was Ironclad Systems headquartered in before Denver?",
     "expected_answer": "Seattle", "as_of": None,
     "fact_ids": ["f050", "f051"], "question_type": "predecessor", "difficulty": 3},

    {"id": "q067", "question": "Who was the CEO of Pegasus Cloud before James Nakamura?",
     "expected_answer": "Diana Osei", "as_of": None,
     "fact_ids": ["f056", "f057"], "question_type": "predecessor", "difficulty": 3},

    {"id": "q068", "question": "Who was the CTO of Helix Technologies before Arthur Flynn?",
     "expected_answer": "Isabella Rossi", "as_of": None,
     "fact_ids": ["f077", "f078"], "question_type": "predecessor", "difficulty": 3},

    {"id": "q069", "question": "Who was the CTO of Helix Technologies before Isabella Rossi?",
     "expected_answer": "Morris Chen", "as_of": None,
     "fact_ids": ["f076", "f077"], "question_type": "predecessor", "difficulty": 3},

    {"id": "q070", "question": "Who was Trident SaaS' CEO before Cyrus Ahmadi?",
     "expected_answer": "Nora Fleming", "as_of": None,
     "fact_ids": ["f081", "f082"], "question_type": "predecessor", "difficulty": 3},

    {"id": "q071", "question": "Who was Cobalt Compute's CEO before Maya Gupta?",
     "expected_answer": "Ray Johansson", "as_of": None,
     "fact_ids": ["f089", "f090"], "question_type": "predecessor", "difficulty": 3},

    {"id": "q072", "question": "What was Cobalt Compute's primary cloud partner before AWS?",
     "expected_answer": "GCP", "as_of": None,
     "fact_ids": ["f091", "f092"], "question_type": "predecessor", "difficulty": 3},

    {"id": "q073", "question": "Who was VertexOps' CEO before Leo Brennan?",
     "expected_answer": "Sarah McAllister", "as_of": None,
     "fact_ids": ["f093", "f094"], "question_type": "predecessor", "difficulty": 3},

    {"id": "q074", "question": "Who was PineCore's CEO before Anaya Patel?",
     "expected_answer": "Thomas Berger", "as_of": None,
     "fact_ids": ["f097", "f098"], "question_type": "predecessor", "difficulty": 3},

    {"id": "q075", "question": "Who was Dawnlight AI's CEO before Elena Novak?",
     "expected_answer": "Victor Osei", "as_of": None,
     "fact_ids": ["f101", "f102"], "question_type": "predecessor", "difficulty": 3},

    # ===== transition (25) =====
    {"id": "q076", "question": "When did Velox Systems change its CEO from Marta Osei to Derek Holt?",
     "expected_answer": "2023-08-15", "as_of": None,
     "fact_ids": ["f001", "f002"], "question_type": "transition", "difficulty": 3},

    {"id": "q077", "question": "When did Velox Systems move its headquarters from Austin to San Francisco?",
     "expected_answer": "2024-06-01", "as_of": None,
     "fact_ids": ["f004", "f005"], "question_type": "transition", "difficulty": 3},

    {"id": "q078", "question": "When did Orbita Analytics appoint Leila Vasquez as CEO?",
     "expected_answer": "2024-02-28", "as_of": None,
     "fact_ids": ["f006", "f007"], "question_type": "transition", "difficulty": 3},

    {"id": "q079", "question": "When did Orbita Analytics raise its Series C?",
     "expected_answer": "2024-07-01", "as_of": None,
     "fact_ids": ["f009", "f010"], "question_type": "transition", "difficulty": 3},

    {"id": "q080", "question": "When did NimbusDrive appoint Aisha Okafor as CEO?",
     "expected_answer": "2022-03-01", "as_of": None,
     "fact_ids": ["f011", "f012"], "question_type": "transition", "difficulty": 3},

    {"id": "q081", "question": "When did Crestline AI replace Tobias Meier with Sandra Liu as CEO?",
     "expected_answer": "2024-09-30", "as_of": None,
     "fact_ids": ["f016", "f017"], "question_type": "transition", "difficulty": 3},

    {"id": "q082", "question": "When did PrismaFlow launch FlowCore v3?",
     "expected_answer": "2025-02-01", "as_of": None,
     "fact_ids": ["f020", "f021"], "question_type": "transition", "difficulty": 3},

    {"id": "q083", "question": "When did QuantumLeap Software promote Marcus Reid to CEO?",
     "expected_answer": "2023-12-01", "as_of": None,
     "fact_ids": ["f024", "f025"], "question_type": "transition", "difficulty": 3},

    {"id": "q084", "question": "When did Zenith Platforms relocate its headquarters from London to Berlin?",
     "expected_answer": "2023-09-01", "as_of": None,
     "fact_ids": ["f030", "f031"], "question_type": "transition", "difficulty": 3},

    {"id": "q085", "question": "When did Apex Infra change its CEO from Rachel Schwartz to Dimitri Volkov?",
     "expected_answer": "2024-01-01", "as_of": None,
     "fact_ids": ["f032", "f033"], "question_type": "transition", "difficulty": 3},

    {"id": "q086", "question": "When did TerraStack switch its primary cloud partner from AWS to Google Cloud?",
     "expected_answer": "2024-03-01", "as_of": None,
     "fact_ids": ["f038", "f039"], "question_type": "transition", "difficulty": 3},

    {"id": "q087", "question": "When did Luminary Data appoint Tyler Grant as CEO?",
     "expected_answer": "2025-04-15", "as_of": None,
     "fact_ids": ["f040", "f041"], "question_type": "transition", "difficulty": 3},

    {"id": "q088", "question": "When did Cascade Networks transition from Fatima Al-Hassan to Marcus Bell as CEO?",
     "expected_answer": "2023-11-30", "as_of": None,
     "fact_ids": ["f044", "f045"], "question_type": "transition", "difficulty": 3},

    {"id": "q089", "question": "When did Ironclad Systems move its headquarters from Seattle to Denver?",
     "expected_answer": "2023-01-15", "as_of": None,
     "fact_ids": ["f050", "f051"], "question_type": "transition", "difficulty": 3},

    {"id": "q090", "question": "When did SkyBridge Tech launch BridgeOS 2.0?",
     "expected_answer": "2024-09-01", "as_of": None,
     "fact_ids": ["f054", "f055"], "question_type": "transition", "difficulty": 3},

    {"id": "q091", "question": "When did Solaris DevTools relocate from Portland to Austin?",
     "expected_answer": "2024-07-01", "as_of": None,
     "fact_ids": ["f063", "f064"], "question_type": "transition", "difficulty": 3},

    {"id": "q092", "question": "When did Helix Technologies appoint Arthur Flynn as CTO?",
     "expected_answer": "2024-06-15", "as_of": None,
     "fact_ids": ["f077", "f078"], "question_type": "transition", "difficulty": 3},

    {"id": "q093", "question": "When did Meridian Labs switch its primary cloud partner from Azure to AWS?",
     "expected_answer": "2023-08-01", "as_of": None,
     "fact_ids": ["f071", "f072"], "question_type": "transition", "difficulty": 3},

    {"id": "q094", "question": "When did CoralByte appoint Hannah White as CEO?",
     "expected_answer": "2024-11-01", "as_of": None,
     "fact_ids": ["f073", "f074"], "question_type": "transition", "difficulty": 3},

    {"id": "q095", "question": "When did Nimbus Edge relocate its headquarters from Tokyo to Singapore?",
     "expected_answer": "2025-01-01", "as_of": None,
     "fact_ids": ["f087", "f088"], "question_type": "transition", "difficulty": 3},

    {"id": "q096", "question": "When did Cobalt Compute change CEO from Ray Johansson to Maya Gupta?",
     "expected_answer": "2020-10-01", "as_of": None,
     "fact_ids": ["f089", "f090"], "question_type": "transition", "difficulty": 3},

    {"id": "q097", "question": "When did VertexOps launch OpsHub 2.0?",
     "expected_answer": "2023-07-15", "as_of": None,
     "fact_ids": ["f095", "f096"], "question_type": "transition", "difficulty": 3},

    {"id": "q098", "question": "When did PineCore move its headquarters from Toronto to New York?",
     "expected_answer": "2023-04-15", "as_of": None,
     "fact_ids": ["f099", "f100"], "question_type": "transition", "difficulty": 3},

    {"id": "q099", "question": "When did StormPath Analytics appoint Rosa Martinez as CEO?",
     "expected_answer": "2025-02-28", "as_of": None,
     "fact_ids": ["f105", "f106"], "question_type": "transition", "difficulty": 3},

    {"id": "q100", "question": "When did BrightGrid Solutions appoint Ji-Yeon Park as CTO?",
     "expected_answer": "2024-05-15", "as_of": None,
     "fact_ids": ["f108", "f109"], "question_type": "transition", "difficulty": 3},
]


def main() -> None:
    corpus_path = HERE / "corpus.jsonl"
    questions_path = HERE / "questions.jsonl"

    with corpus_path.open("w") as f:
        for fact in CORPUS:
            f.write(json.dumps(fact) + "\n")
    print(f"Wrote {len(CORPUS)} facts to {corpus_path}")

    with questions_path.open("w") as f:
        for q in QUESTIONS:
            f.write(json.dumps(q) + "\n")
    print(f"Wrote {len(QUESTIONS)} questions to {questions_path}")


if __name__ == "__main__":
    main()
