"""
Intelli-Credit: Fraud Graph Analytics
Uses NetworkX to detect hidden corporate fraud networks.
"""

import os
from typing import Dict, List, Any
from dataclasses import dataclass, field, asdict

import numpy as np

try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False


@dataclass
class FraudRiskReport:
    company_id: str
    company_name: str
    overall_risk_score: float
    risk_level: str
    shell_company_flags: List[Dict] = field(default_factory=list)
    circular_trading_flags: List[Dict] = field(default_factory=list)
    shared_director_clusters: List[Dict] = field(default_factory=list)
    promoter_network_risks: List[Dict] = field(default_factory=list)
    centrality_scores: Dict[str, float] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)


class FraudGraphAnalyzer:
    """NetworkX corporate fraud network analysis."""

    def __init__(self):
        if not HAS_NX:
            raise ImportError("networkx required")
        self.graph = nx.DiGraph()
        self._company_data = {}

    def build_graph(self, companies: List[Dict], relationships: List[Dict]) -> None:
        self.graph.clear()
        for c in companies:
            self.graph.add_node(c["id"], name=c.get("name", ""), node_type=c.get("type", "company"),
                                revenue_cr=c.get("revenue_cr", 0), employees=c.get("employees", 0),
                                gst_turnover_cr=c.get("gst_turnover_cr", 0))
            self._company_data[c["id"]] = c
        for r in relationships:
            self.graph.add_edge(r["source"], r["target"], edge_type=r.get("type", "related"),
                                weight=r.get("weight", 1.0))

    def analyze_company(self, company_id: str) -> FraudRiskReport:
        name = self._company_data.get(company_id, {}).get("name", company_id)
        shells = self.detect_shell_companies(company_id)
        circular = self.detect_circular_trading(company_id)
        clusters = self.find_shared_director_clusters(company_id)
        promoter_risks = self.analyze_promoter_network(company_id)
        centrality = self.compute_centrality(company_id)

        score = min(10.0, len(shells)*1.5 + len(circular)*2.0 +
                    sum(1 for c in clusters if c["cluster_size"]>=5) +
                    sum(1.5 for p in promoter_risks if p["risk_level"]=="HIGH") +
                    (1.0 if centrality.get("betweenness",0)>0.15 else 0))

        level = "CRITICAL" if score>=8 else "HIGH" if score>=6 else "MEDIUM" if score>=4 else "LOW"
        recs = self._recommendations(level, shells, circular, clusters, promoter_risks)

        return FraudRiskReport(company_id=company_id, company_name=name,
                               overall_risk_score=round(score,1), risk_level=level,
                               shell_company_flags=shells, circular_trading_flags=circular,
                               shared_director_clusters=clusters, promoter_network_risks=promoter_risks,
                               centrality_scores=centrality, recommendations=recs)

    def detect_shell_companies(self, company_id: str) -> List[Dict]:
        flags = []
        if company_id not in self.graph:
            return flags
        connected = set()
        for n in nx.all_neighbors(self.graph, company_id):
            connected.add(n)
            for n2 in nx.all_neighbors(self.graph, n):
                connected.add(n2)

        for nid in connected:
            if nid == company_id:
                continue
            nd = self.graph.nodes.get(nid, {})
            if nd.get("node_type") != "company":
                continue
            rev = nd.get("revenue_cr", 0)
            emp = nd.get("employees", 0)
            deg = self.graph.degree(nid)
            reasons = []
            if rev < 0.5 and deg >= 3:
                reasons.append(f"Low revenue ₹{rev:.1f}Cr with {deg} connections")
            if emp < 5 and deg >= 3:
                reasons.append(f"Only {emp} employees with {deg} connections")
            if reasons:
                flags.append({"entity_id": nid, "entity_name": nd.get("name",""),
                              "shell_probability": min(0.9, 0.3+0.1*deg), "reasons": reasons})
        return flags

    def detect_circular_trading(self, company_id: str) -> List[Dict]:
        flags = []
        if company_id not in self.graph:
            return flags
        trade_edges = [(u,v) for u,v,d in self.graph.edges(data=True)
                       if d.get("edge_type") in ("gst_trade","related_party","trade")]
        if not trade_edges:
            return flags
        tg = nx.DiGraph(trade_edges)
        if company_id not in tg:
            return flags
        try:
            for cycle in nx.simple_cycles(tg):
                if company_id in cycle and len(cycle) <= 5:
                    names = [self.graph.nodes.get(n,{}).get("name",n) for n in cycle]
                    flags.append({"cycle": cycle, "cycle_names": names, "cycle_length": len(cycle),
                                  "risk": "HIGH" if len(cycle)<=3 else "MEDIUM",
                                  "description": f"Circular: {' → '.join(names)} → {names[0]}"})
                    if len(flags) >= 10:
                        break
        except Exception:
            pass
        return flags

    def find_shared_director_clusters(self, company_id: str) -> List[Dict]:
        clusters = []
        if company_id not in self.graph:
            return clusters
        directors = set()
        for n in list(self.graph.predecessors(company_id)) + list(self.graph.successors(company_id)):
            if self.graph.nodes.get(n, {}).get("node_type") == "director":
                directors.add(n)
        for did in directors:
            cos = [n for n in nx.all_neighbors(self.graph, did)
                   if self.graph.nodes.get(n,{}).get("node_type")=="company"]
            if len(cos) > 1:
                clusters.append({"director_id": did, "director_name": self.graph.nodes.get(did,{}).get("name",""),
                                  "companies": cos, "cluster_size": len(cos),
                                  "risk": "HIGH" if len(cos)>=5 else "MEDIUM" if len(cos)>=3 else "LOW"})
        return clusters

    def analyze_promoter_network(self, company_id: str) -> List[Dict]:
        risks = []
        if company_id not in self.graph:
            return risks
        promoters = [n for n in nx.all_neighbors(self.graph, company_id)
                     if self.graph.nodes.get(n,{}).get("node_type")=="promoter"]
        for pid in promoters:
            cos = [n for n in nx.all_neighbors(self.graph, pid)
                   if self.graph.nodes.get(n,{}).get("node_type")=="company"]
            factors = []
            if len(cos) >= 5:
                factors.append(f"Connected to {len(cos)} companies")
            if len(cos) >= 3:
                defaulted = [c for c in cos if self._company_data.get(c,{}).get("defaulted",False)]
                if defaulted:
                    factors.append(f"Linked to {len(defaulted)} defaulted entities")
            if factors:
                risks.append({"promoter_id": pid, "promoter_name": self.graph.nodes.get(pid,{}).get("name",""),
                               "connected_companies": len(cos), "risk_factors": factors,
                               "risk_level": "HIGH" if len(factors)>=2 else "MEDIUM"})
        return risks

    def compute_centrality(self, company_id: str) -> Dict[str, float]:
        if company_id not in self.graph or len(self.graph) < 2:
            return {"degree": 0, "betweenness": 0, "pagerank": 0, "clustering": 0}
        try:
            return {
                "degree": round(float(nx.degree_centrality(self.graph).get(company_id, 0)), 4),
                "betweenness": round(float(nx.betweenness_centrality(self.graph).get(company_id, 0)), 4),
                "pagerank": round(float(nx.pagerank(self.graph).get(company_id, 0)), 4),
                "clustering": round(float(nx.clustering(self.graph.to_undirected()).get(company_id, 0)), 4),
            }
        except Exception:
            return {"degree": 0, "betweenness": 0, "pagerank": 0, "clustering": 0}

    def _recommendations(self, level, shells, circular, clusters, promoter) -> List[str]:
        recs = []
        if level in ("HIGH","CRITICAL"):
            recs.append("Mandatory enhanced due diligence required")
        if shells:
            recs.append("Investigate potential shell companies for fund siphoning risk")
        if circular:
            recs.append("Investigate circular trading — potential GST fraud or revenue inflation")
        if any(c["cluster_size"]>=5 for c in clusters):
            recs.append("Verify rationale for directors serving on 5+ boards")
        if any(p["risk_level"]=="HIGH" for p in promoter):
            recs.append("Deep-dive into promoter's other ventures")
        if not recs:
            recs.append("Standard due diligence sufficient — no elevated fraud risk")
        return recs

    def build_sample_graph(self, company_name: str = "Target Company") -> FraudRiskReport:
        companies = [
            {"id":"target","name":company_name,"type":"company","revenue_cr":25,"employees":120},
            {"id":"s1","name":"ABC Suppliers","type":"company","revenue_cr":8,"employees":30},
            {"id":"s2","name":"XYZ Traders","type":"company","revenue_cr":0.3,"employees":2},
            {"id":"c1","name":"PQR Industries","type":"company","revenue_cr":50,"employees":200},
            {"id":"shell","name":"Shell Corp","type":"company","revenue_cr":0.1,"employees":1},
            {"id":"p1","name":"Rajesh Kumar","type":"promoter"},
            {"id":"p2","name":"Suresh Patel","type":"promoter"},
            {"id":"d1","name":"Amit Shah","type":"director"},
            {"id":"d2","name":"Priya Mehta","type":"director"},
        ]
        relationships = [
            {"source":"p1","target":"target","type":"promoter_of"},
            {"source":"p1","target":"shell","type":"promoter_of"},
            {"source":"p1","target":"s2","type":"promoter_of"},
            {"source":"p2","target":"target","type":"promoter_of"},
            {"source":"d1","target":"target","type":"director_of"},
            {"source":"d1","target":"s1","type":"director_of"},
            {"source":"d1","target":"c1","type":"director_of"},
            {"source":"d2","target":"target","type":"director_of"},
            {"source":"d2","target":"shell","type":"director_of"},
            {"source":"target","target":"s1","type":"gst_trade"},
            {"source":"s1","target":"s2","type":"gst_trade"},
            {"source":"s2","target":"target","type":"gst_trade"},
            {"source":"target","target":"c1","type":"gst_trade"},
            {"source":"shell","target":"target","type":"related_party"},
        ]
        self.build_graph(companies, relationships)
        return self.analyze_company("target")
