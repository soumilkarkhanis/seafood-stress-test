import streamlit as st
import pandas as pd
import numpy as np
import pickle
import gdown
import os
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY

# ============================================================
# LOAD BACKEND FROM GOOGLE DRIVE
# ============================================================

@st.cache_resource
def load_backend():
    # Download files if not present
    if not os.path.exists("meta.pkl"):
        gdown.download("https://drive.google.com/uc?id=1nZP8SYKlZiwr3TgoYvR0DFDoS0DKiY7S", "meta.pkl", quiet=False)
    if not os.path.exists("W_only.pkl"):
        gdown.download("https://drive.google.com/uc?id=1_EbT78iLlc055BJSVZd_9yVhOnmC6T6Y", "W_only.pkl", quiet=False)

    with open("meta.pkl", "rb") as f:
        meta = pickle.load(f)
    with open("W_only.pkl", "rb") as f:
        W_np = pickle.load(f)

    st.info("Building network matrices... this takes about 2 minutes on first load.")

    countries = meta["GLOBAL_COUNTRIES"]
    idx = {c: i for i, c in enumerate(countries)}

    # Recompute W2, W3, E as numpy float32
    W_species = W_np
    W2_species = {}
    W3_species = {}
    E_species = {}

    TRANSMISSION_MODES = meta["TRANSMISSION_MODES"]

    for sp, W in W_species.items():
        W2 = W @ W
        W3 = W2 @ W
        W2_species[sp] = W2
        W3_species[sp] = W3
        E_species[sp] = {
            mode: W + p["w2"] * W2 + p["w3"] * W3
            for mode, p in TRANSMISSION_MODES.items()
        }

    return {
        "GLOBAL_COUNTRIES": countries,
        "species_list": meta["species_list"],
        "W_species": W_species,
        "W2_species": W2_species,
        "W3_species": W3_species,
        "E_species": E_species,
        "country_roles": meta["country_roles"],
        "export_capacity_species": meta["export_capacity_species"],
        "TRANSMISSION_MODES": TRANSMISSION_MODES,
    }

backend = load_backend()

GLOBAL_COUNTRIES        = backend["GLOBAL_COUNTRIES"]
W_species               = backend["W_species"]
W2_species              = backend["W2_species"]
W3_species              = backend["W3_species"]
E_species               = backend["E_species"]
country_roles           = backend["country_roles"]
export_capacity_species = backend["export_capacity_species"]
TRANSMISSION_MODES      = backend["TRANSMISSION_MODES"]
ALL_COUNTRIES_SORTED    = sorted(GLOBAL_COUNTRIES)

# ============================================================
# LOOKUPS
# ============================================================

ISO3_TO_NAME = {
    'AFG':'Afghanistan','AGO':'Angola','ALB':'Albania','ARE':'United Arab Emirates',
    'ARG':'Argentina','ARM':'Armenia','AUS':'Australia','AUT':'Austria','AZE':'Azerbaijan',
    'BDI':'Burundi','BEL':'Belgium','BEN':'Benin','BGD':'Bangladesh','BGR':'Bulgaria',
    'BHR':'Bahrain','BHS':'Bahamas','BIH':'Bosnia and Herzegovina','BLR':'Belarus',
    'BLZ':'Belize','BOL':'Bolivia','BRA':'Brazil','BTN':'Bhutan','BWA':'Botswana',
    'CAN':'Canada','CHE':'Switzerland','CHL':'Chile','CHN':'China','CIV':'Ivory Coast',
    'CMR':'Cameroon','COD':'DR Congo','COG':'Congo','COL':'Colombia','CRI':'Costa Rica',
    'CUB':'Cuba','CYP':'Cyprus','CZE':'Czech Republic','DEU':'Germany','DNK':'Denmark',
    'DOM':'Dominican Republic','DZA':'Algeria','ECU':'Ecuador','EGY':'Egypt','ESP':'Spain',
    'EST':'Estonia','ETH':'Ethiopia','FIN':'Finland','FJI':'Fiji','FRA':'France',
    'GAB':'Gabon','GBR':'United Kingdom','GEO':'Georgia','GHA':'Ghana','GIN':'Guinea',
    'GMB':'Gambia','GRC':'Greece','GTM':'Guatemala','HND':'Honduras','HRV':'Croatia',
    'HTI':'Haiti','HUN':'Hungary','IDN':'Indonesia','IND':'India','IRL':'Ireland',
    'IRN':'Iran','IRQ':'Iraq','ISL':'Iceland','ISR':'Israel','ITA':'Italy',
    'JAM':'Jamaica','JOR':'Jordan','JPN':'Japan','KAZ':'Kazakhstan','KEN':'Kenya',
    'KHM':'Cambodia','KOR':'South Korea','KWT':'Kuwait','LAO':'Laos','LBN':'Lebanon',
    'LBR':'Liberia','LBY':'Libya','LKA':'Sri Lanka','LTU':'Lithuania','LUX':'Luxembourg',
    'LVA':'Latvia','MAR':'Morocco','MDA':'Moldova','MDG':'Madagascar','MDV':'Maldives',
    'MEX':'Mexico','MLI':'Mali','MLT':'Malta','MMR':'Myanmar','MOZ':'Mozambique',
    'MRT':'Mauritania','MUS':'Mauritius','MWI':'Malawi','MYS':'Malaysia','NAM':'Namibia',
    'NER':'Niger','NGA':'Nigeria','NIC':'Nicaragua','NLD':'Netherlands','NOR':'Norway',
    'NPL':'Nepal','NZL':'New Zealand','OMN':'Oman','PAK':'Pakistan','PAN':'Panama',
    'PER':'Peru','PHL':'Philippines','PNG':'Papua New Guinea','POL':'Poland',
    'PRK':'North Korea','PRT':'Portugal','PRY':'Paraguay','QAT':'Qatar','ROU':'Romania',
    'RUS':'Russia','RWA':'Rwanda','SAU':'Saudi Arabia','SDN':'Sudan','SEN':'Senegal',
    'SGP':'Singapore','SLE':'Sierra Leone','SLV':'El Salvador','SOM':'Somalia',
    'SRB':'Serbia','SSD':'South Sudan','SUR':'Suriname','SVK':'Slovakia','SVN':'Slovenia',
    'SWE':'Sweden','SWZ':'Eswatini','SYR':'Syria','TCD':'Chad','TGO':'Togo',
    'THA':'Thailand','TJK':'Tajikistan','TKM':'Turkmenistan','TLS':'Timor-Leste',
    'TON':'Tonga','TTO':'Trinidad and Tobago','TUN':'Tunisia','TUR':'Turkey',
    'TZA':'Tanzania','UGA':'Uganda','UKR':'Ukraine','URY':'Uruguay','USA':'United States',
    'UZB':'Uzbekistan','VEN':'Venezuela','VNM':'Vietnam','YEM':'Yemen',
    'ZAF':'South Africa','ZMB':'Zambia','ZWE':'Zimbabwe','MKD':'North Macedonia',
    'MNE':'Montenegro','BFA':'Burkina Faso','KGZ':'Kyrgyzstan','KIR':'Kiribati',
    'LSO':'Lesotho','MNG':'Mongolia','ATG':'Antigua and Barbuda','BRB':'Barbados',
    'BRN':'Brunei','COM':'Comoros','GUY':'Guyana','PSE':'Palestine',
    'GNB':'Guinea-Bissau','GNQ':'Equatorial Guinea','CAF':'Central African Republic',
    'CPV':'Cape Verde','NFK':'Norfolk Island','STP':'Sao Tome and Principe',
}

COMMON_NAMES = {
    'abramis': 'Bream','abramis brama': 'Common Bream',
    'acanthocybium solandri': 'Wahoo','aequipecten opercularis': 'Queen Scallop',
    'clupea harengus': 'Atlantic Herring','gadus morhua': 'Atlantic Cod',
    'salmo salar': 'Atlantic Salmon','thunnus albacares': 'Yellowfin Tuna',
    'thunnus thynnus': 'Atlantic Bluefin Tuna','thunnus obesus': 'Bigeye Tuna',
    'katsuwonus pelamis': 'Skipjack Tuna','xiphias gladius': 'Swordfish',
    'merluccius': 'Hake','gadus chalcogrammus': 'Walleye Pollock',
    'pangasianodon hypophthalmus': 'Pangasius','oreochromis niloticus': 'Nile Tilapia',
    'penaeus vannamei': 'Whiteleg Shrimp','penaeus monodon': 'Giant Tiger Prawn',
    'chionoecetes opilio': 'Snow Crab','paralithodes camtschaticus': 'Red King Crab',
    'homarus americanus': 'American Lobster','pandalus borealis': 'Northern Shrimp',
    'mytilus edulis': 'Blue Mussel','ostrea edulis': 'European Oyster',
    'lates niloticus': 'Nile Perch','dicentrarchus labrax': 'European Seabass',
    'sparus aurata': 'Gilthead Seabream','oncorhynchus mykiss': 'Rainbow Trout',
    'oncorhynchus nerka': 'Sockeye Salmon','scomber scombrus': 'Atlantic Mackerel',
    'nephrops norvegicus': 'Norway Lobster','octopus vulgaris': 'Common Octopus',
    'engraulis encrasicolus': 'European Anchovy','sardina pilchardus': 'European Sardine',
    'melanogrammus aeglefinus': 'Haddock','micromesistius poutassou': 'Blue Whiting',
}

def c_name(code):
    return ISO3_TO_NAME.get(code, code)

def s_name(sciname):
    return COMMON_NAMES.get(sciname.lower(), sciname.title())

def display_species(sp):
    common = COMMON_NAMES.get(sp.lower())
    return f"{common} ({sp})" if common else sp.title()

def sci_from_display(display):
    if '(' in display:
        return display.split('(')[-1].rstrip(')')
    return display.lower()

# ============================================================
# BACKEND FUNCTIONS
# ============================================================

def build_shock_vector(shocks, countries):
    s = np.zeros(len(countries))
    idx = {c: i for i, c in enumerate(countries)}
    for c, v in shocks.items():
        if c in idx:
            s[idx[c]] = v
    return s

def country_exposure(species, focal_country, shocks):
    W  = W_species[species]
    E_modes = E_species[species]
    countries = GLOBAL_COUNTRIES
    idx = {c: i for i, c in enumerate(countries)}
    s = build_shock_vector(shocks, countries)
    i = idx[focal_country]
    direct = float(W[i] @ s)
    results = {}
    for mode, E in E_modes.items():
        results[mode] = float(E[i] @ s)
    low  = results["Low transmission"]
    mod  = results["Moderate transmission"]
    full = results["Full transmission"]
    return {
        "species": species, "focal_country": focal_country,
        "shock_scenario": shocks, "direct_exposure": direct,
        "low_transmission_exposure": low,
        "moderate_transmission_exposure": mod,
        "full_transmission_exposure": full,
        "network_exposure_min": min(low, mod, full),
        "network_exposure_max": max(low, mod, full),
        "amplification_low": low - direct,
        "amplification_moderate": mod - direct,
        "amplification_full": full - direct,
    }

def firm_exposure(species, sourcing_vector, shocks):
    W  = W_species[species]
    W2 = W2_species[species]
    countries = GLOBAL_COUNTRIES
    idx = {c: i for i, c in enumerate(countries)}
    f = np.zeros(len(countries))
    for c, v in sourcing_vector.items():
        if c in idx:
            f[idx[c]] = v
    if f.sum() <= 0:
        raise ValueError("Sourcing vector sums to zero.")
    f = f / f.sum()
    s = build_shock_vector(shocks, countries)
    direct = float(f @ s)
    results = {}
    for mode, params in TRANSMISSION_MODES.items():
        results[mode] = float(f @ s) + params["w2"] * float(f @ W @ s) + params["w3"] * float(f @ W2 @ s)
    low  = results["Low transmission"]
    mod  = results["Moderate transmission"]
    full = results["Full transmission"]
    return {
        "species": species, "direct_exposure": direct,
        "low_transmission_exposure": low,
        "moderate_transmission_exposure": mod,
        "full_transmission_exposure": full,
        "network_exposure_min": min(low, mod, full),
        "network_exposure_max": max(low, mod, full),
        "amplification_low": low - direct,
        "amplification_moderate": mod - direct,
        "amplification_full": full - direct,
    }

def exposure_decomposition(species, focal_country, shocks):
    W = W_species[species]
    E_modes = E_species[species]
    countries = GLOBAL_COUNTRIES
    idx = {c: i for i, c in enumerate(countries)}
    i = idx[focal_country]
    rows = []
    for shock_country, shock_value in shocks.items():
        if shock_country not in idx:
            continue
        j = idx[shock_country]
        direct_c = float(W[i, j] * shock_value)
        row = {"shock_country": shock_country, "shock_value": shock_value,
               "direct_contribution": direct_c}
        for mode, E in E_modes.items():
            c = float(E[i, j] * shock_value)
            row[f"{mode}_contribution"] = c
            row[f"{mode}_amplification"] = c - direct_c
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    decomp = pd.DataFrame(rows)
    return decomp.sort_values("Moderate transmission_contribution", ascending=False).reset_index(drop=True)

def supplier_scores(species, shocks, transmission_mode="Moderate transmission", min_export_share=0.001):
    W = W_species[species]
    E = E_species[species][transmission_mode]
    countries = GLOBAL_COUNTRIES
    s = build_shock_vector(shocks, countries)
    T = E @ s
    direct_shock = s.copy()
    supplier_risk = np.maximum(direct_shock, T).clip(0, 1)
    exp_cap = export_capacity_species.get(species)
    if exp_cap is None:
        return pd.DataFrame()
    export_share = np.array([exp_cap.get(c, 0) for c in countries], dtype=float)
    scores = pd.DataFrame({
        "country": countries,
        "export_share": export_share,
        "supplier_risk": supplier_risk,
    })
    scores["supplier_score"] = scores["export_share"] * (1 - scores["supplier_risk"])
    return scores[scores["export_share"] >= min_export_share].sort_values(
        "supplier_score", ascending=False).reset_index(drop=True)

def recommend_alternative_suppliers(species, focal_country, shocks,
                                     transmission_mode="Moderate transmission",
                                     top_n=8, min_export_share=0.001,
                                     exclude_current_suppliers=False,
                                     exclude_shocked_countries=True):
    countries = GLOBAL_COUNTRIES
    idx = {c: i for i, c in enumerate(countries)}
    scores = supplier_scores(species, shocks, transmission_mode, min_export_share)
    if scores.empty:
        return scores
    scores = scores[scores["country"] != focal_country]
    if exclude_shocked_countries:
        scores = scores[~scores["country"].isin(shocks.keys())]
    if exclude_current_suppliers and focal_country in idx:
        i = idx[focal_country]
        W = W_species[species]
        current = [countries[j] for j in range(len(countries)) if W[i, j] > 0]
        scores = scores[~scores["country"].isin(current)]
    if scores.empty:
        return scores
    eq75 = scores["export_share"].quantile(0.75)
    rq25 = scores["supplier_risk"].quantile(0.25)
    rq50 = scores["supplier_risk"].quantile(0.50)
    def reason(row):
        r = "High export capacity" if row["export_share"] >= eq75 else "Meaningful export capacity"
        if row["supplier_risk"] <= rq25:   r += "; low shock exposure"
        elif row["supplier_risk"] <= rq50: r += "; moderate shock exposure"
        else:                              r += "; moderate indirect exposure"
        return r
    scores["recommendation_reason"] = scores.apply(reason, axis=1)
    return scores.head(top_n).reset_index(drop=True)

def exposure_from_profile(species, sourcing_profile, shocks):
    W  = W_species[species]
    W2 = W2_species[species]
    countries = GLOBAL_COUNTRIES
    idx = {c: i for i, c in enumerate(countries)}
    f = np.array([sourcing_profile.get(c, 0) for c in countries], dtype=float)
    if f.sum() <= 0:
        raise ValueError("Sourcing profile sums to zero.")
    f = f / f.sum()
    s = build_shock_vector(shocks, countries)
    direct = float(f @ s)
    results = {}
    for mode, params in TRANSMISSION_MODES.items():
        results[mode] = float(f @ s) + params["w2"] * float(f @ W @ s) + params["w3"] * float(f @ W2 @ s)
    low = results["Low transmission"]
    mod = results["Moderate transmission"]
    full = results["Full transmission"]
    return {
        "direct_exposure": direct,
        "low_transmission_exposure": low,
        "moderate_transmission_exposure": mod,
        "full_transmission_exposure": full,
        "network_exposure_min": min(results.values()),
        "network_exposure_max": max(results.values()),
        "amplification_low": low - direct,
        "amplification_moderate": mod - direct,
        "amplification_full": full - direct,
    }

def create_diversified_profile(species, focal_country, shocks,
                                shift_share=0.20, transmission_mode="Moderate transmission",
                                min_export_share=0.001):
    countries = GLOBAL_COUNTRIES
    idx = {c: i for i, c in enumerate(countries)}
    W = W_species[species]
    i = idx[focal_country]
    current_profile = {c: float(W[i, j]) for j, c in enumerate(countries) if W[i, j] > 0}
    total = sum(current_profile.values())
    if total == 0:
        raise ValueError(f"{focal_country} is not an active importer for {species}.")
    current_profile = {c: v/total for c, v in current_profile.items()}
    decomp = exposure_decomposition(species, focal_country, shocks)
    if decomp.empty:
        raise ValueError("No decomposition available.")
    reduce_from = [c for c in decomp["shock_country"].tolist() if current_profile.get(c, 0) > 0]
    if not reduce_from:
        raise ValueError("No current suppliers overlap with shocked countries.")
    alts = recommend_alternative_suppliers(species, focal_country, shocks,
                                           transmission_mode, top_n=5,
                                           min_export_share=min_export_share,
                                           exclude_current_suppliers=False,
                                           exclude_shocked_countries=True)
    alts = alts[~alts["country"].isin(reduce_from)]
    add_to = alts["country"].tolist()
    if not add_to:
        raise ValueError("No alternative suppliers found.")
    diversified = dict(current_profile)
    reduce_total = sum(diversified.get(c, 0) for c in reduce_from)
    actual_shift = min(shift_share, reduce_total)
    red_w = {c: diversified.get(c, 0) / reduce_total for c in reduce_from}
    for c in reduce_from:
        diversified[c] = diversified.get(c, 0) - actual_shift * red_w[c]
    sc = supplier_scores(species, shocks, transmission_mode, min_export_share)
    sc = sc[sc["country"].isin(add_to)]
    total_score = sc["supplier_score"].sum()
    add_w = {row["country"]: row["supplier_score"]/total_score for _, row in sc.iterrows()} if total_score > 0 else {c: 1/len(add_to) for c in add_to}
    for c in add_to:
        diversified[c] = diversified.get(c, 0) + actual_shift * add_w.get(c, 0)
    diversified = {c: max(0, v) for c, v in diversified.items()}
    total_div = sum(diversified.values())
    diversified = {c: v/total_div for c, v in diversified.items()}
    return diversified, {"reduced_from": reduce_from, "added_to": add_to, "actual_shift_share": actual_shift}

def compare_current_vs_diversified(species, focal_country, shocks,
                                    shift_share=0.20, transmission_mode="Moderate transmission",
                                    min_export_share=0.001):
    countries = GLOBAL_COUNTRIES
    idx = {c: i for i, c in enumerate(countries)}
    W = W_species[species]
    i = idx[focal_country]
    current_profile = {c: float(W[i, j]) for j, c in enumerate(countries)}
    total = sum(current_profile.values())
    current_profile = {c: v/total for c, v in current_profile.items() if total > 0}
    div_profile, plan = create_diversified_profile(species, focal_country, shocks, shift_share, transmission_mode, min_export_share)
    cur_exp = exposure_from_profile(species, current_profile, shocks)
    div_exp = exposure_from_profile(species, div_profile, shocks)
    comparison = pd.DataFrame([
        {"portfolio": "Current sourcing", **cur_exp},
        {"portfolio": "Diversified sourcing", **div_exp},
    ])
    return comparison, plan, div_profile

def pct(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "N/A"
    return f"{100 * x:.1f}%"

def exposure_label(val):
    if val is None: return "Unknown"
    if val < 0.05:  return "Low"
    if val < 0.15:  return "Moderate"
    if val < 0.30:  return "High"
    return "Severe"

# ============================================================
# PDF
# ============================================================

def make_pdf_report(species, focal_country, shocks, exposure_result,
                    decomp_df, alts_df, comparison_df, profile_change_df, plan,
                    mode="Country Importer", sourcing_vector=None):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                            leftMargin=2.5*cm, rightMargin=2.5*cm,
                            topMargin=2.5*cm, bottomMargin=2.5*cm)
    BASE = getSampleStyleSheet()
    def S(name, parent="Normal", **kwargs):
        return ParagraphStyle(name, parent=BASE[parent], **kwargs)
    h1 = S("H1","Heading1",fontSize=14,textColor=colors.HexColor("#0d3d6e"),
            fontName="Helvetica-Bold",spaceBefore=18,spaceAfter=6)
    h2 = S("H2","Heading2",fontSize=11,textColor=colors.HexColor("#1a5c9a"),
            fontName="Helvetica-Bold",spaceBefore=12,spaceAfter=4)
    body = S("Body","Normal",fontSize=9.5,leading=15,alignment=TA_JUSTIFY,
             textColor=colors.HexColor("#222222"),spaceAfter=8)
    body_small = S("BodySmall","Normal",fontSize=8.5,leading=13,
                   textColor=colors.HexColor("#444444"),spaceAfter=6)
    callout = S("Callout","Normal",fontSize=10,leading=16,
                textColor=colors.HexColor("#0d3d6e"),
                backColor=colors.HexColor("#eef4fb"),
                borderPad=8,spaceAfter=10,leftIndent=12,rightIndent=12,
                fontName="Helvetica-Bold")
    caveat_style = S("Caveat","Normal",fontSize=8,leading=12,
                     textColor=colors.HexColor("#666666"))
    tbl_hdr = colors.HexColor("#0d3d6e")
    tbl_alt = colors.HexColor("#f0f5fb")
    def hr():
        return HRFlowable(width="100%",thickness=0.5,
                          color=colors.HexColor("#c8d8ea"),spaceAfter=8,spaceBefore=4)
    def tbl_style():
        return TableStyle([
            ("BACKGROUND",(0,0),(-1,0),tbl_hdr),
            ("TEXTCOLOR",(0,0),(-1,0),colors.white),
            ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
            ("FONTSIZE",(0,0),(-1,-1),8),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,tbl_alt]),
            ("GRID",(0,0),(-1,-1),0.3,colors.HexColor("#c0cfe0")),
            ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
            ("TOPPADDING",(0,0),(-1,-1),5),
            ("BOTTOMPADDING",(0,0),(-1,-1),5),
            ("LEFTPADDING",(0,0),(-1,-1),6),
        ])
    story = []
    story.append(Spacer(1,1.5*cm))
    story.append(Paragraph("Seafood Supply Chain Stress-Testing Report",
                           S("T","Title",fontSize=22,textColor=colors.HexColor("#0d3d6e"),
                             fontName="Helvetica-Bold")))
    story.append(Spacer(1,0.3*cm))
    story.append(hr())
    mode_label = "Country-Level Analysis" if mode=="Country Importer" else "Firm-Level Analysis"
    story.append(Paragraph(f"Analysis type: {mode_label}",
                           S("cs","Normal",fontSize=12,textColor=colors.HexColor("#3a6ea5"),spaceAfter=4)))
    story.append(Paragraph(f"Species: {s_name(species)}",
                           S("cs2","Normal",fontSize=12,textColor=colors.HexColor("#3a6ea5"),spaceAfter=4)))
    story.append(Paragraph(f"Subject: {c_name(focal_country) if mode=='Country Importer' else focal_country}",
                           S("cs3","Normal",fontSize=12,textColor=colors.HexColor("#3a6ea5"),spaceAfter=4)))
    shock_str = ", ".join([f"{c_name(c)} ({int(v*100)}% shock)" for c,v in shocks.items()])
    story.append(Paragraph(f"Shock scenario: {shock_str}",
                           S("cs4","Normal",fontSize=12,textColor=colors.HexColor("#3a6ea5"),spaceAfter=4)))
    story.append(Spacer(1,0.3*cm))
    story.append(Paragraph("Prepared using the Seafood Trade Stress-Testing Tool. Data: 2016-2020 bilateral trade flows. This report is analytical and does not constitute financial or procurement advice.",
                           S("disc","Normal",fontSize=9,textColor=colors.HexColor("#888888"),spaceAfter=2)))
    story.append(Spacer(1,1.5*cm))

    direct  = exposure_result["direct_exposure"]
    mod_exp = exposure_result["moderate_transmission_exposure"]
    exp_min = exposure_result["network_exposure_min"]
    exp_max = exposure_result["network_exposure_max"]
    amp_full= exposure_result["amplification_full"]
    label   = exposure_label(mod_exp)

    story.append(Paragraph("1. Executive Summary", h1))
    story.append(hr())
    if mode=="Country Importer":
        subject_desc = f"the country of {c_name(focal_country)}"
    else:
        top_src = sorted(sourcing_vector.items(), key=lambda x:-x[1])[:3] if sourcing_vector else []
        subject_desc = "a firm sourcing from " + ", ".join([f"{c_name(c)} ({int(v*100)}%)" for c,v in top_src])
    story.append(Paragraph(
        f"This report assesses the seafood supply chain exposure of {subject_desc} "
        f"to a disruption scenario affecting {shock_str}. "
        f"Direct exposure is estimated at <b>{pct(direct)}</b>. "
        f"Network-adjusted exposure rises to between <b>{pct(exp_min)} and {pct(exp_max)}</b> "
        f"(moderate estimate: {pct(mod_exp)}). "
        f"This indicates a <b>{label.lower()} level of procurement risk</b>.", body))
    if amp_full > 0.01:
        amp_ratio = amp_full/direct if direct > 0.001 else 0
        if amp_ratio > 5:
            story.append(Paragraph(
                f"Indirect network effects amplify exposure by {round(amp_ratio)}x beyond direct sourcing links "
                f"({pct(direct)} direct vs {pct(mod_exp)} network-adjusted), indicating significant upstream dependency.",body))
    if mod_exp < 0.005 and direct < 0.005:
        bottom = "Exposure is negligible under this scenario. The focal importer does not source meaningfully from shocked countries, either directly or through upstream network pathways."
    elif not comparison_df.empty:
        div_mod = comparison_df.loc[1,"moderate_transmission_exposure"]
        from_str = ", ".join([c_name(c) for c in plan.get("reduced_from",[])])
        to_str   = ", ".join([c_name(c) for c in plan.get("added_to",[])])
        bottom = f"Shifting {pct(plan.get('actual_shift_share',0.2))} of sourcing away from {from_str} toward {to_str} reduces exposure from {pct(mod_exp)} to {pct(div_mod)}."
    else:
        bottom = f"Moderate-transmission exposure is {pct(mod_exp)} ({label}). {'Sourcing appears structurally resilient.' if mod_exp < 0.05 else 'Sourcing concentration warrants review.'}"
    story.append(Paragraph(f"Key finding: {bottom}", callout))

    story.append(Paragraph("2. Exposure Analysis", h1))
    story.append(hr())
    story.append(Paragraph("2.1 What this scenario represents", h2))
    story.append(Paragraph(
        f"The analysis applies a supply disruption to {len(shocks)} "
        f"{'country' if len(shocks)==1 else 'countries'}: {shock_str}. "
        f"A shock reduces the affected country's effective export capacity by the stated percentage. "
        f"This could represent sanctions, a climate event, a disease outbreak, or a logistics disruption.",body))
    story.append(Paragraph("2.2 Direct vs. network exposure", h2))
    story.append(Paragraph(
        f"<b>Direct exposure ({pct(direct)})</b> measures sourcing that flows directly from shocked countries. "
        f"<b>Network-adjusted exposure ({pct(exp_min)}-{pct(exp_max)})</b> incorporates indirect dependencies — "
        f"your suppliers' suppliers. The range reflects low, moderate, and full transmission assumptions.",body))
    story.append(Paragraph("These estimates indicate structural vulnerability, not precise supply shortfall or price predictions.",body_small))
    data = [["Metric","Value","Interpretation"],
            ["Direct exposure",pct(direct),"Sourcing directly from shocked countries"],
            ["Low transmission",pct(exposure_result['low_transmission_exposure']),"Network effects at 25%/10% of W2/W3"],
            ["Moderate transmission",pct(mod_exp),"Central estimate"],
            ["Full transmission",pct(exposure_result['full_transmission_exposure']),"Upper bound"],
            ["Network amplification",f"{pct(exposure_result['amplification_low'])} - {pct(amp_full)}","Additional indirect exposure"]]
    t = Table(data,colWidths=[5*cm,3*cm,8.5*cm])
    t.setStyle(tbl_style())
    story.append(t)
    story.append(Spacer(1,0.4*cm))

    if not decomp_df.empty:
        story.append(Paragraph("3. Risk Drivers", h1))
        story.append(hr())
        story.append(Paragraph("Which shocked countries contribute most to exposure, and how much is direct vs. network-mediated?",body))
        d = decomp_df[["shock_country","shock_value","direct_contribution","Moderate transmission_contribution","Moderate transmission_amplification"]].copy()
        d["shock_country"] = d["shock_country"].apply(c_name)
        d.columns = ["Country","Shock","Direct","Network (moderate)","Amplification"]
        for col in ["Shock","Direct","Network (moderate)","Amplification"]:
            d[col] = d[col].apply(pct)
        data = [list(d.columns)] + d.values.tolist()
        t = Table(data,colWidths=[3*cm,2.5*cm,3.5*cm,3.5*cm,4*cm])
        t.setStyle(tbl_style())
        story.append(t)
        story.append(Spacer(1,0.3*cm))
        top = decomp_df.iloc[0]
        if top["Moderate transmission_contribution"] > 0.005:
            story.append(Paragraph(f"The dominant risk driver is <b>{c_name(top['shock_country'])}</b>, contributing {pct(top['Moderate transmission_contribution'])} of total exposure under moderate transmission.",body))
        else:
            story.append(Paragraph("No shocked country contributes meaningfully to exposure. The focal importer does not appear to source from shocked countries directly or through upstream intermediaries.",body))

    if not alts_df.empty:
        story.append(Paragraph("4. Alternative Supplier Recommendations", h1))
        story.append(hr())
        story.append(Paragraph("Countries ranked by export capacity weighted against their own shock exposure. A supplier exposed to the same shock scores lower.",body))
        a = alts_df[["country","export_share","supplier_risk","supplier_score","recommendation_reason"]].copy()
        a["country"] = a["country"].apply(c_name)
        a.columns = ["Country","Export share","Supplier risk","Score","Rationale"]
        a["Export share"] = a["Export share"].apply(pct)
        a["Supplier risk"] = a["Supplier risk"].apply(pct)
        a["Score"] = a["Score"].apply(lambda x: f"{x:.3f}")
        data = [list(a.columns)] + a.values.tolist()
        t = Table(data,colWidths=[3*cm,3*cm,3*cm,2.5*cm,5*cm])
        t.setStyle(tbl_style())
        story.append(t)
        story.append(Spacer(1,0.3*cm))
        top_alt = alts_df.iloc[0]
        story.append(Paragraph(f"<b>{c_name(top_alt['country'])}</b> is the top-ranked alternative, holding {pct(top_alt['export_share'])} of global exports while remaining relatively insulated from the shock.",body))

    if comparison_df is not None and not comparison_df.empty:
        story.append(Paragraph("5. Diversification Scenario", h1))
        story.append(hr())
        from_str = ", ".join([c_name(c) for c in plan.get("reduced_from",[])])
        to_str   = ", ".join([c_name(c) for c in plan.get("added_to",[])])
        shift    = plan.get("actual_shift_share",0.2)
        story.append(Paragraph(f"Shifting {pct(shift)} of procurement away from <b>{from_str}</b> toward <b>{to_str}</b>.",body))
        cur = comparison_df.loc[0]
        div = comparison_df.loc[1]
        comp_data = [["Portfolio","Direct","Low","Moderate","Full","Interval"],
                     ["Current",pct(cur["direct_exposure"]),pct(cur["low_transmission_exposure"]),
                      pct(cur["moderate_transmission_exposure"]),pct(cur["full_transmission_exposure"]),
                      f"{pct(cur['network_exposure_min'])}-{pct(cur['network_exposure_max'])}"],
                     ["Diversified",pct(div["direct_exposure"]),pct(div["low_transmission_exposure"]),
                      pct(div["moderate_transmission_exposure"]),pct(div["full_transmission_exposure"]),
                      f"{pct(div['network_exposure_min'])}-{pct(div['network_exposure_max'])}"]]
        t = Table(comp_data,colWidths=[3*cm,2.5*cm,2.5*cm,2.5*cm,2.5*cm,3.5*cm])
        t.setStyle(tbl_style())
        story.append(t)
        story.append(Spacer(1,0.3*cm))
        red = cur["moderate_transmission_exposure"] - div["moderate_transmission_exposure"]
        story.append(Paragraph(f"Diversification reduces moderate-estimate exposure by <b>{pct(red)}</b> — from {pct(cur['moderate_transmission_exposure'])} to {pct(div['moderate_transmission_exposure'])}.",body))
        if not profile_change_df.empty:
            story.append(Paragraph("Supplier share changes:",h2))
            data = [["Country","Current share","Diversified share","Change"]]
            for _,r in profile_change_df.iterrows():
                chg = f"+{pct(r['change'])}" if r["change"]>0 else pct(r["change"])
                data.append([c_name(r["country"]),pct(r["current_share"]),pct(r["diversified_share"]),chg])
            t = Table(data,colWidths=[4*cm,4*cm,4.5*cm,4*cm])
            t.setStyle(tbl_style())
            story.append(t)

    story.append(Paragraph("6. Methodology", h1))
    story.append(hr())
    story.append(Paragraph("6.1 Data",h2))
    story.append(Paragraph("Bilateral seafood trade flows, 192 countries, 2016-2020. Recency-weighted average (weights 1-5). Importer-species pairs below 10 tonnes excluded.",body))
    story.append(Paragraph("6.2 Network model",h2))
    story.append(Paragraph("Truncated Leontief expansion: W (direct), W2, W3 (indirect). Three transmission matrices: E_low=W+0.25W2+0.10W3, E_moderate=W+0.50W2+0.25W3, E_full=W+W2+W3.",body))
    story.append(Paragraph("6.3 Limitations",h2))
    story.append(Paragraph("Structural exposure estimate only — not price or volume predictions. Data end 2020. Transmission coefficients are assumptions not empirical estimates.",caveat_style))
    doc.build(story)
    buffer.seek(0)
    return buffer

# ============================================================
# PAGE CONFIG & STYLES
# ============================================================

st.set_page_config(page_title="Seafood Supply Chain Risk Assessment",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@400;600&family=DM+Sans:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; background-color: #f7f9fc; color: #1a2332; }
.stApp { background-color: #f7f9fc; }
.stSidebar { background-color: #ffffff !important; border-right: 1px solid #dde4ee !important; }
.stSidebar, .stSidebar * { font-family: 'DM Sans', sans-serif !important; color: #1a2332 !important; }
.stButton > button { background-color: #0d3d6e; color: #ffffff !important; border: none; border-radius: 3px; padding: 10px 20px; font-family: 'DM Sans', sans-serif; font-weight: 500; font-size: 14px; width: 100%; margin-top: 12px; }
.stButton > button:hover { background-color: #1a5c9a; }
.stButton > button p { color: #ffffff !important; }
.stNumberInput input { background-color: #ffffff !important; color: #1a2332 !important; border: 1px solid #c8d4e4 !important; }
.stNumberInput button { background-color: #eef2f7 !important; color: #1a2332 !important; }
.stNumberInput button svg { fill: #1a2332 !important; }
[data-baseweb="select"] * { color: #1a2332 !important; background-color: #ffffff !important; }
.stRadio label { color: #1a2332 !important; }
.metric-card { background: white; border: 1px solid #dde4ee; border-top: 3px solid #0d3d6e; border-radius: 4px; padding: 20px 24px; margin-bottom: 12px; }
.metric-value { font-family: 'Source Serif 4', serif; font-size: 32px; font-weight: 600; color: #0d3d6e; margin: 4px 0; }
.metric-label { font-size: 11px; text-transform: uppercase; letter-spacing: 1.2px; color: #7a8fa8; font-weight: 500; margin-bottom: 4px; }
.metric-sublabel { font-size: 12px; color: #5a7a9a; margin-top: 4px; line-height: 1.5; }
.metric-badge { display: inline-block; font-size: 11px; font-weight: 600; padding: 2px 8px; border-radius: 2px; margin-top: 6px; }
.badge-low { background: #e8f5ee; color: #2a7a4b; }
.badge-moderate { background: #fef5e7; color: #b07d2a; }
.badge-high { background: #fef0e6; color: #c05a1a; }
.badge-severe { background: #fde8e8; color: #a02020; }
.section-rule { border: none; border-top: 1px solid #dde4ee; margin: 28px 0 20px 0; }
.section-title { font-family: 'Source Serif 4', serif; font-size: 18px; font-weight: 600; color: #0d3d6e; margin-bottom: 4px; }
.section-desc { font-size: 13px; color: #6a8099; margin-bottom: 20px; line-height: 1.5; }
.recommendation-panel { background: #f0f6f0; border-left: 4px solid #2a7a4b; padding: 16px 20px; border-radius: 0 4px 4px 0; margin: 12px 0 20px 0; font-size: 14px; color: #1a3a2a; line-height: 1.7; }
.warning-panel { background: #fef8f0; border-left: 4px solid #b07d2a; padding: 16px 20px; border-radius: 0 4px 4px 0; margin: 12px 0 20px 0; font-size: 14px; color: #3a2a0a; line-height: 1.7; }
.info-note { background: #f0f4f8; border: 1px solid #c8d4e4; border-radius: 4px; padding: 10px 14px; font-size: 12px; color: #5a7a9a; margin-top: 8px; line-height: 1.6; }
.sidebar-section { font-size: 11px; text-transform: uppercase; letter-spacing: 1.2px; color: #7a8fa8; font-weight: 600; margin-top: 20px; margin-bottom: 8px; padding-bottom: 4px; border-bottom: 1px solid #eef2f7; }
.landing-card { background: white; border: 1px solid #dde4ee; border-radius: 4px; padding: 24px; }
.landing-step { font-size: 11px; text-transform: uppercase; letter-spacing: 1px; color: #0d3d6e; font-weight: 600; margin-bottom: 8px; }
.landing-text { font-size: 13px; color: #5a7a9a; line-height: 1.6; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div style="border-bottom:2px solid #0d3d6e;padding-bottom:20px;margin-bottom:28px;text-align:center;">
    <div style="font-family:'Source Serif 4',serif;font-size:26px;font-weight:600;color:#0d3d6e;">
        Seafood Supply Chain Risk Assessment
    </div>
    <div style="font-size:13px;color:#7a8fa8;margin-top:6px;line-height:1.6;">
        Network stress-testing tool &nbsp;·&nbsp; Indirect exposure analysis &nbsp;·&nbsp; 192 countries &nbsp;·&nbsp; 1,300+ species
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown('<div class="sidebar-section">Analysis Mode</div>', unsafe_allow_html=True)
    mode = st.radio("", ["Country Importer", "Firm / Custom Sourcing"], label_visibility="collapsed")

    st.markdown('<div class="sidebar-section">Species</div>', unsafe_allow_html=True)
    _species_options = sorted([display_species(sp) for sp in W_species.keys()])
    _species_display = st.selectbox("", _species_options, label_visibility="collapsed")
    species = sci_from_display(_species_display)

    st.markdown('<div class="sidebar-section">Focus</div>', unsafe_allow_html=True)
    if mode == "Country Importer":
        _fc_options = [f"{c_name(c)} ({c})" for c in ALL_COUNTRIES_SORTED]
        _fc_display = st.selectbox("Importing country", _fc_options)
        focal_country = _fc_display.split("(")[-1].rstrip(")")
        firm_shares = {}
    else:
        st.caption("Countries you source from")
        _firm_opts = [f"{c_name(c)} ({c})" for c in ALL_COUNTRIES_SORTED]
        _firm_sel  = st.multiselect("", _firm_opts, label_visibility="collapsed")
        firm_shares = {}
        if _firm_sel:
            st.caption("Sourcing share per country (must sum to 100%)")
            total_share = 0.0
            for item in _firm_sel:
                code  = item.split("(")[-1].rstrip(")")
                share = st.number_input(f"{item} (%)", min_value=0.0, max_value=100.0,
                                        value=round(100/len(_firm_sel),1), step=0.5, key=f"share_{code}")
                firm_shares[code] = share
                total_share += share
            if firm_shares:
                if abs(total_share - 100) > 0.5:
                    st.warning(f"Total: {total_share:.1f}% — must equal 100%")
                else:
                    st.success(f"Total: {total_share:.1f}%")
        focal_country = list(firm_shares.keys())[0] if firm_shares else None

    st.markdown('<div class="sidebar-section">Shock Scenario</div>', unsafe_allow_html=True)
    st.caption("Select countries experiencing a supply disruption")
    _shock_opts = [f"{c_name(c)} ({c})" for c in ALL_COUNTRIES_SORTED]
    _shock_sel  = st.multiselect("", _shock_opts, label_visibility="collapsed")
    shock_countries = [d.split("(")[-1].rstrip(")") for d in _shock_sel]

    shock_dict = {}
    if shock_countries:
        for c in shock_countries:
            preset = st.radio(f"{c_name(c)}", ["Mild (10%)", "Moderate (25%)", "Severe (40%)", "Custom"],
                              index=1, horizontal=True, key=f"preset_{c}")
            if preset == "Mild (10%)":       level = 0.10
            elif preset == "Moderate (25%)": level = 0.25
            elif preset == "Severe (40%)":   level = 0.40
            else:
                level = st.number_input(f"Custom % for {c_name(c)}", min_value=1, max_value=99,
                                        value=25, step=1, key=f"custom_{c}") / 100
            shock_dict[c] = level

    st.markdown('<div class="sidebar-section">Diversification</div>', unsafe_allow_html=True)
    st.caption("Share of total sourcing to reallocate away from high-risk suppliers")
    shift_share = st.select_slider("", options=[0.10, 0.20, 0.30], value=0.20,
                                   format_func=lambda x: f"{int(x*100)}% of total sourcing",
                                   label_visibility="collapsed")
    st.markdown("---")
    run = st.button("Run Stress Test")

# ============================================================
# LANDING
# ============================================================

if not run:
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        st.markdown("""
        <div style="text-align:center;padding:32px 0 24px 0;">
            <div style="font-family:'Source Serif 4',serif;font-size:24px;font-weight:600;color:#0d3d6e;margin-bottom:12px;line-height:1.4;">
                Standard procurement risk assessments stop at your direct suppliers.<br>Supply shocks rarely do.
            </div>
            <div style="font-size:14px;color:#3a5060;line-height:1.8;margin-bottom:28px;">
                A sanctions regime, an export ban, a climate event — these propagate through
                upstream trade networks far beyond the countries you buy from directly.
                This tool maps your exposure through three layers of the global seafood
                trade network: 192 countries, 1,300+ species, 2016-2020 bilateral flows.
            </div>
        </div>
        <div style="background:white;border:1px solid #dde4ee;border-radius:6px;padding:20px 24px;margin-bottom:12px;display:flex;gap:16px;align-items:flex-start;">
            <div style="background:#eef4fb;border-radius:50%;width:32px;height:32px;min-width:32px;display:flex;align-items:center;justify-content:center;font-family:'Source Serif 4',serif;font-weight:600;color:#0d3d6e;font-size:15px;">1</div>
            <div>
                <div style="font-weight:600;color:#1a2332;font-size:14px;margin-bottom:4px;">Choose your species and analysis mode</div>
                <div style="font-size:13px;color:#6a8099;line-height:1.6;">Select country-level analysis to benchmark against a national import profile, or firm mode to enter your own sourcing shares directly.</div>
            </div>
        </div>
        <div style="background:white;border:1px solid #dde4ee;border-radius:6px;padding:20px 24px;margin-bottom:12px;display:flex;gap:16px;align-items:flex-start;">
            <div style="background:#eef4fb;border-radius:50%;width:32px;height:32px;min-width:32px;display:flex;align-items:center;justify-content:center;font-family:'Source Serif 4',serif;font-weight:600;color:#0d3d6e;font-size:15px;">2</div>
            <div>
                <div style="font-weight:600;color:#1a2332;font-size:14px;margin-bottom:4px;">Define a disruption scenario</div>
                <div style="font-size:13px;color:#6a8099;line-height:1.6;">Select any country or countries experiencing a disruption. Set severity from mild to severe, or enter a custom percentage. You don't need to source directly from them for it to affect you.</div>
            </div>
        </div>
        <div style="background:white;border:1px solid #dde4ee;border-radius:6px;padding:20px 24px;margin-bottom:24px;display:flex;gap:16px;align-items:flex-start;">
            <div style="background:#eef4fb;border-radius:50%;width:32px;height:32px;min-width:32px;display:flex;align-items:center;justify-content:center;font-family:'Source Serif 4',serif;font-weight:600;color:#0d3d6e;font-size:15px;">3</div>
            <div>
                <div style="font-weight:600;color:#1a2332;font-size:14px;margin-bottom:4px;">Run the analysis and explore results</div>
                <div style="font-size:13px;color:#6a8099;line-height:1.6;">Get your direct and network-adjusted exposure, identify risk drivers, explore alternative suppliers, and simulate a sourcing reallocation. Download a full PDF report.</div>
            </div>
        </div>
        <div style="background:#f0f4f8;border:1px solid #c8d4e4;border-radius:4px;padding:14px 18px;font-size:12px;color:#5a7a9a;line-height:1.7;">
            <b style="color:#1a2332;">Built on real trade data.</b>
            Bilateral seafood trade flows across 192 countries (2016-2020), modelling up to three layers of upstream dependency using network propagation. Covers 1,315 species. Results indicate structural vulnerability — not price movements or guaranteed supply shortfalls.
        </div>
        """, unsafe_allow_html=True)
    st.stop()

# ============================================================
# VALIDATION
# ============================================================

if not shock_dict:
    st.markdown('<div class="warning-panel">Please select at least one shock country in the sidebar.</div>', unsafe_allow_html=True)
    st.stop()

if mode == "Firm / Custom Sourcing":
    if not firm_shares:
        st.markdown('<div class="warning-panel">Please enter your sourcing countries and shares.</div>', unsafe_allow_html=True)
        st.stop()
    if abs(sum(firm_shares.values()) - 100) > 0.5:
        st.markdown('<div class="warning-panel">Sourcing shares must sum to 100%.</div>', unsafe_allow_html=True)
        st.stop()

if mode == "Country Importer":
    idx_map = {c: i for i, c in enumerate(GLOBAL_COUNTRIES)}
    if focal_country not in idx_map:
        st.markdown('<div class="warning-panel">Selected country not found in trade data.</div>', unsafe_allow_html=True)
        st.stop()

# ============================================================
# RUN ANALYSIS
# ============================================================

with st.spinner("Running analysis..."):
    if mode == "Country Importer":
        is_active = focal_country in country_roles.get(species, {}).get("active_importers", [])
        if not is_active:
            st.markdown(f'<div class="warning-panel"><b>Note:</b> {c_name(focal_country)} does not appear as an active importer of {s_name(species)} in the 2016-2020 data.</div>', unsafe_allow_html=True)
        exposure_result = country_exposure(species, focal_country, shock_dict)
        decomp_df = exposure_decomposition(species, focal_country, shock_dict)
        alts_df = recommend_alternative_suppliers(species, focal_country, shock_dict,
                                                   top_n=8, min_export_share=0.001,
                                                   exclude_current_suppliers=False,
                                                   exclude_shocked_countries=True)
        if not alts_df.empty:
            _current_w = W_species[species]
            _i = {c: i for i, c in enumerate(GLOBAL_COUNTRIES)}[focal_country]
            _current_set = set(GLOBAL_COUNTRIES[j] for j in range(192) if _current_w[_i, j] > 0)
            alts_df["current_supplier"] = alts_df["country"].apply(lambda x: "Yes" if x in _current_set else "No")
        try:
            comparison_df, plan, diversified_profile = compare_current_vs_diversified(
                species, focal_country, shock_dict, shift_share=shift_share)
            _curr = {c: float(W_species[species][{c2: i2 for i2, c2 in enumerate(GLOBAL_COUNTRIES)}[focal_country], j])
                     for j, c in enumerate(GLOBAL_COUNTRIES)}
            _curr_total = sum(_curr.values())
            _curr = {c: v/_curr_total for c, v in _curr.items() if _curr_total > 0}
            all_c = sorted(set(_curr.keys()) | set(diversified_profile.keys()))
            profile_change_df = pd.DataFrame([{
                "country": c,
                "current_share": _curr.get(c, 0),
                "diversified_share": diversified_profile.get(c, 0),
                "change": diversified_profile.get(c, 0) - _curr.get(c, 0),
                "abs_change": abs(diversified_profile.get(c, 0) - _curr.get(c, 0))
            } for c in all_c if _curr.get(c, 0) > 0 or diversified_profile.get(c, 0) > 0])
            profile_change_df = profile_change_df.sort_values("abs_change", ascending=False).head(15).reset_index(drop=True)
        except Exception as e:
            comparison_df = pd.DataFrame()
            plan = {}
            diversified_profile = {}
            profile_change_df = pd.DataFrame()
        sourcing_vector_for_pdf = None
    else:
        sourcing_vector = {c: v/100 for c, v in firm_shares.items()}
        exposure_result = firm_exposure(species, sourcing_vector, shock_dict)
        decomp_df = pd.DataFrame()
        alts_df = recommend_alternative_suppliers(species, list(sourcing_vector.keys())[0],
                                                   shock_dict, top_n=8,
                                                   exclude_current_suppliers=False,
                                                   exclude_shocked_countries=True)
        comparison_df = pd.DataFrame()
        plan = {}
        profile_change_df = pd.DataFrame()
        sourcing_vector_for_pdf = sourcing_vector

# ============================================================
# RESULTS
# ============================================================

direct   = exposure_result["direct_exposure"]
mod_exp  = exposure_result["moderate_transmission_exposure"]
exp_min  = exposure_result["network_exposure_min"]
exp_max  = exposure_result["network_exposure_max"]
amp_low  = exposure_result["amplification_low"]
amp_full = exposure_result["amplification_full"]
label    = exposure_label(mod_exp)
badge_class = f"badge-{label.lower()}"

st.markdown('<hr class="section-rule">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Exposure Summary</div>', unsafe_allow_html=True)
st.markdown('<div class="section-desc">Direct exposure reflects sourcing from shocked countries. Network-adjusted exposure accounts for indirect upstream dependencies.</div>', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Direct Exposure</div><div class="metric-value">{pct(direct)}</div><div class="metric-sublabel">Share of sourcing from directly shocked countries</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Moderate Estimate</div><div class="metric-value">{pct(mod_exp)}</div><div class="metric-sublabel">Central network-adjusted estimate</div><span class="metric-badge {badge_class}">{label}</span></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Exposure Interval</div><div class="metric-value" style="font-size:22px;">{pct(exp_min)} – {pct(exp_max)}</div><div class="metric-sublabel">Across low, moderate, and full transmission assumptions</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Network Amplification</div><div class="metric-value" style="font-size:22px;">{pct(amp_low)} – {pct(amp_full)}</div><div class="metric-sublabel">Additional exposure from indirect upstream dependencies</div></div>', unsafe_allow_html=True)

st.markdown('<hr class="section-rule">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Assessment</div>', unsafe_allow_html=True)

if not comparison_df.empty:
    cur_min  = comparison_df.loc[0,"network_exposure_min"]
    cur_max  = comparison_df.loc[0,"network_exposure_max"]
    div_min  = comparison_df.loc[1,"network_exposure_min"]
    div_max  = comparison_df.loc[1,"network_exposure_max"]
    from_str = ", ".join([c_name(c) for c in plan.get("reduced_from",[])])
    to_str   = ", ".join([c_name(c) for c in plan.get("added_to",[])])
    st.markdown(f'<div class="recommendation-panel">Under this shock scenario, exposure ranges from <b>{pct(cur_min)} to {pct(cur_max)}</b> (moderate estimate: <b>{pct(mod_exp)}</b> — rated <b>{label}</b>). Shifting {int(shift_share*100)}% of sourcing away from <b>{from_str}</b> toward <b>{to_str}</b> reduces the exposure interval to <b>{pct(div_min)}–{pct(div_max)}</b>.</div>', unsafe_allow_html=True)
else:
    panel = "recommendation-panel" if mod_exp < 0.20 else "warning-panel"
    msg   = "Sourcing appears structurally resilient under this scenario." if mod_exp < 0.10 else "Sourcing concentration warrants review. See alternative supplier recommendations below."
    st.markdown(f'<div class="{panel}">Moderate-transmission exposure is <b>{pct(mod_exp)}</b> — rated <b>{label}</b>. {msg}</div>', unsafe_allow_html=True)

st.markdown('<hr class="section-rule">', unsafe_allow_html=True)
left, right = st.columns(2)

with left:
    st.markdown('<div class="section-title">Risk Drivers</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Which shocked countries contribute most, and how much is direct vs. network-mediated?</div>', unsafe_allow_html=True)
    if not decomp_df.empty:
        d = decomp_df[["shock_country","shock_value","direct_contribution","Moderate transmission_contribution"]].copy()
        d["shock_country"] = d["shock_country"].apply(c_name)
        d.columns = ["Country","Shock","Direct","Network (Moderate)"]
        for col in ["Shock","Direct","Network (Moderate)"]:
            d[col] = d[col].apply(pct)
        st.dataframe(d, use_container_width=True, hide_index=True)
    else:
        st.markdown('<div class="info-note">Risk decomposition available in Country Importer mode only.</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="section-title">Alternative Suppliers</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Ranked by export capacity weighted against shock resilience.</div>', unsafe_allow_html=True)
    if not alts_df.empty:
        cols = ["country","export_share","supplier_risk","supplier_score","recommendation_reason"]
        if "current_supplier" in alts_df.columns:
            cols.append("current_supplier")
        a = alts_df[cols].copy()
        a["country"] = a["country"].apply(c_name)
        a["export_share"]  = a["export_share"].apply(pct)
        a["supplier_risk"] = a["supplier_risk"].apply(pct)
        a["supplier_score"] = a["supplier_score"].apply(lambda x: f"{x:.3f}")
        new_cols = ["Country","Export share","Supplier risk","Score","Rationale"]
        if "current_supplier" in alts_df.columns:
            new_cols.append("Currently sourcing")
        a.columns = new_cols
        st.dataframe(a, use_container_width=True, hide_index=True)
    else:
        st.markdown('<div class="info-note">No alternative suppliers found.</div>', unsafe_allow_html=True)

if not comparison_df.empty:
    st.markdown('<hr class="section-rule">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Diversification Scenario</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-desc">Impact of shifting {int(shift_share*100)}% of sourcing away from high-risk suppliers toward recommended alternatives.</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    cur = comparison_df.loc[0]
    div = comparison_df.loc[1]
    with c1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Current Sourcing</div><div class="metric-value">{pct(cur["moderate_transmission_exposure"])}</div><div class="metric-sublabel">Interval: {pct(cur["network_exposure_min"])} – {pct(cur["network_exposure_max"])}</div></div>', unsafe_allow_html=True)
    with c2:
        div_label = exposure_label(div["moderate_transmission_exposure"])
        st.markdown(f'<div class="metric-card"><div class="metric-label">After Diversification</div><div class="metric-value">{pct(div["moderate_transmission_exposure"])}</div><div class="metric-sublabel">Interval: {pct(div["network_exposure_min"])} – {pct(div["network_exposure_max"])}</div><span class="metric-badge badge-{div_label.lower()}">{div_label}</span></div>', unsafe_allow_html=True)
    if not profile_change_df.empty:
        st.markdown("**Supplier share changes**")
        pcd = profile_change_df.copy()
        pcd["country"] = pcd["country"].apply(c_name)
        pcd["current_share"]     = pcd["current_share"].apply(pct)
        pcd["diversified_share"] = pcd["diversified_share"].apply(pct)
        pcd["change"]            = pcd["change"].apply(lambda x: f"+{pct(x)}" if x > 0 else pct(x))
        pcd = pcd.drop(columns=["abs_change"], errors="ignore")
        st.dataframe(pcd, use_container_width=True, hide_index=True)

st.markdown('<hr class="section-rule">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Download Report</div>', unsafe_allow_html=True)
st.markdown('<div class="section-desc">Full consulting-style PDF with methodology and plain-language interpretation.</div>', unsafe_allow_html=True)

focal_for_pdf = focal_country if mode == "Country Importer" else "Firm (custom sourcing)"
pdf_buffer = make_pdf_report(
    species=species, focal_country=focal_for_pdf, shocks=shock_dict,
    exposure_result=exposure_result, decomp_df=decomp_df, alts_df=alts_df,
    comparison_df=comparison_df, profile_change_df=profile_change_df,
    plan=plan, mode=mode, sourcing_vector=sourcing_vector_for_pdf)

st.download_button(label="Download PDF Report", data=pdf_buffer,
                   file_name="seafood_stress_test_report.pdf",
                   mime="application/pdf", use_container_width=True)

st.markdown("""
<div style="margin-top:48px;padding-top:16px;border-top:1px solid #dde4ee;color:#aab4c0;font-size:11px;text-align:center;">
    Seafood Supply Chain Risk Assessment &nbsp;·&nbsp; Trade data 2016-2020 &nbsp;·&nbsp;
    Exposure estimates are structural indicators, not price or volume forecasts.
</div>
""", unsafe_allow_html=True)
