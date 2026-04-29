"""Sample datasets for testing topic modeling approaches."""

import pandas as pd
import numpy as np


def get_sample_datasets() -> dict:
    """Return available sample datasets with descriptions."""
    return {
        "20 Newsgroups (subset)": "Classic text classification dataset with newsgroup posts across multiple topics",
        "UN General Debate Speeches (synthetic)": "Synthetic UN-style speeches with timestamps and country metadata for STM",
        "Research Abstracts (synthetic)": "Synthetic research paper abstracts across multiple disciplines",
    }


def load_sample_dataset(name: str) -> pd.DataFrame:
    """Load a sample dataset by name."""
    if name == "20 Newsgroups (subset)":
        return _load_newsgroups()
    elif name == "UN General Debate Speeches (synthetic)":
        return _load_un_speeches()
    elif name == "Research Abstracts (synthetic)":
        return _load_research_abstracts()
    else:
        raise ValueError(f"Unknown dataset: {name}")


def _load_newsgroups() -> pd.DataFrame:
    """Load a subset of the 20 Newsgroups dataset."""
    try:
        from sklearn.datasets import fetch_20newsgroups
        categories = [
            "sci.space", "sci.med", "comp.graphics", "talk.politics.mideast",
            "rec.sport.baseball", "soc.religion.christian"
        ]
        data = fetch_20newsgroups(
            subset="train", categories=categories,
            remove=("headers", "footers", "quotes"), random_state=42
        )
        df = pd.DataFrame({
            "text": data.data,
            "category": [data.target_names[t] for t in data.target]
        })
        # Clean up
        df["text"] = df["text"].str.strip()
        df = df[df["text"].str.len() > 50].reset_index(drop=True)
        # Take a manageable subset
        if len(df) > 500:
            df = df.groupby("category", group_keys=False).apply(
                lambda x: x.sample(min(80, len(x)), random_state=42)
            ).reset_index(drop=True)
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to load 20 Newsgroups: {e}")


def _load_un_speeches() -> pd.DataFrame:
    """Generate synthetic UN General Debate-style speeches with metadata."""
    np.random.seed(42)

    topics_content = {
        "climate_change": [
            "Climate change remains the defining challenge of our time. Rising sea levels threaten coastal communities "
            "while extreme weather events devastate agricultural regions. We must commit to reducing carbon emissions "
            "and investing in renewable energy sources to protect future generations.",
            "The Paris Agreement set ambitious targets for emissions reduction, yet many nations struggle to meet their "
            "commitments. Developing countries bear the brunt of climate impacts despite contributing least to the problem. "
            "International climate finance must be scaled up dramatically.",
            "Our forests are burning, our glaciers melting, and our oceans acidifying at unprecedented rates. The scientific "
            "consensus is clear: we must achieve net-zero emissions by 2050. Green technology transfer to developing nations "
            "is essential for a just transition.",
            "Small island developing states face an existential threat from rising seas. Climate adaptation funding remains "
            "woefully inadequate. We call on major emitters to honor their financial pledges and support vulnerable communities.",
        ],
        "peace_security": [
            "Armed conflicts continue to destabilize regions across the globe. Diplomatic solutions must take precedence "
            "over military interventions. The United Nations peacekeeping operations require adequate funding and clear mandates "
            "to effectively protect civilian populations.",
            "Nuclear proliferation poses a grave threat to international security. We must strengthen the Non-Proliferation "
            "Treaty and work toward complete nuclear disarmament. Verification mechanisms need to be robust and universal.",
            "Terrorism and violent extremism continue to undermine peace and stability. Countering radicalization requires "
            "addressing root causes including poverty, inequality, and lack of educational opportunities.",
            "The Security Council must be reformed to reflect contemporary geopolitical realities. Veto power should not "
            "prevent action in the face of mass atrocities. Responsibility to protect remains a moral imperative.",
        ],
        "economic_development": [
            "Sustainable development goals provide a roadmap for eradicating poverty and reducing inequality. Foreign direct "
            "investment in developing countries must be accompanied by technology transfer and capacity building.",
            "The global financial architecture needs fundamental reform. Debt relief for heavily indebted poor countries "
            "is essential to free resources for education, healthcare, and infrastructure development.",
            "Trade barriers disproportionately affect developing nations. A fair multilateral trading system would unlock "
            "economic potential in the Global South and reduce dependency on foreign aid.",
            "Digital transformation offers unprecedented opportunities for economic leapfrogging. However, the digital divide "
            "threatens to widen existing inequalities. Universal access to broadband connectivity is a development priority.",
        ],
        "human_rights": [
            "Human rights are universal, indivisible, and inalienable. No cultural or religious tradition can justify the "
            "suppression of fundamental freedoms. We must hold all nations to the same standards of accountability.",
            "Gender equality remains elusive in many parts of the world. Women's participation in political life, education, "
            "and the economy must be actively promoted and protected through legislation and cultural change.",
            "The rights of refugees and migrants must be upheld regardless of their legal status. The Global Compact on "
            "Migration provides a framework for safe, orderly, and regular migration that benefits all parties.",
            "Freedom of press and expression are under assault worldwide. Journalists face imprisonment, violence, and death "
            "for reporting truth. Democratic institutions depend on a free and independent media.",
        ],
    }

    countries = [
        "Brazil", "India", "Nigeria", "Germany", "Japan", "Mexico", "South Africa",
        "Australia", "Canada", "France", "Kenya", "Indonesia", "Egypt", "Chile",
        "Norway", "Philippines", "Morocco", "Thailand", "Colombia", "New Zealand"
    ]

    years = list(range(2015, 2025))
    rows = []

    for year in years:
        for country in countries:
            topic_key = np.random.choice(list(topics_content.keys()))
            base_text = np.random.choice(topics_content[topic_key])
            # Add some variation
            openers = [
                f"Distinguished delegates, on behalf of {country}, ",
                f"Mr. President, {country} stands before this assembly to affirm that ",
                f"As the representative of {country}, I wish to emphasize that ",
                f"The people of {country} believe firmly that ",
            ]
            text = np.random.choice(openers) + base_text.lower()
            rows.append({
                "text": text,
                "country": country,
                "year": year,
                "region": _get_region(country),
                "topic_label": topic_key,
                "date": f"{year}-09-{np.random.randint(15, 29):02d}",
            })

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


def _get_region(country: str) -> str:
    """Map country to region."""
    regions = {
        "Brazil": "Latin America", "Chile": "Latin America", "Colombia": "Latin America",
        "Mexico": "Latin America", "India": "Asia", "Japan": "Asia", "Indonesia": "Asia",
        "Philippines": "Asia", "Thailand": "Asia", "Nigeria": "Africa", "South Africa": "Africa",
        "Kenya": "Africa", "Egypt": "Africa", "Morocco": "Africa", "Germany": "Europe",
        "France": "Europe", "Norway": "Europe", "Australia": "Oceania",
        "New Zealand": "Oceania", "Canada": "North America",
    }
    return regions.get(country, "Other")


def _load_research_abstracts() -> pd.DataFrame:
    """Generate synthetic research paper abstracts."""
    np.random.seed(42)

    abstracts = {
        "machine_learning": [
            "We propose a novel deep learning architecture for image classification that achieves state-of-the-art "
            "performance on ImageNet benchmarks. Our model uses attention mechanisms combined with residual connections "
            "to capture both local and global features. Experiments show a 3.2% improvement over existing methods.",
            "This paper presents a reinforcement learning approach to robotic manipulation tasks. Our agent learns "
            "dexterous grasping policies through curriculum learning in simulation, then transfers to real-world robots "
            "with minimal fine-tuning. Results demonstrate successful transfer across multiple object categories.",
            "We introduce a transformer-based model for natural language understanding that leverages multi-task learning "
            "across diverse NLP benchmarks. Pre-training on large corpora followed by task-specific fine-tuning yields "
            "significant improvements on question answering, sentiment analysis, and text summarization.",
            "A new federated learning framework is proposed that addresses data heterogeneity across distributed clients. "
            "Our method uses personalized aggregation strategies to maintain model performance while preserving privacy. "
            "Evaluation on healthcare datasets shows improved accuracy compared to standard federated averaging.",
        ],
        "genomics": [
            "Genome-wide association studies reveal novel loci associated with cardiovascular disease risk in diverse "
            "populations. We analyze whole-genome sequencing data from 50,000 participants across five ethnic groups, "
            "identifying 23 previously unreported genetic variants with significant disease associations.",
            "CRISPR-Cas9 gene editing technology enables precise modification of plant genomes for drought resistance. "
            "We demonstrate successful knockout of water-stress sensitivity genes in rice cultivars, resulting in "
            "30% improved yield under water-limited conditions in controlled greenhouse experiments.",
            "Single-cell RNA sequencing reveals heterogeneous cell populations within tumor microenvironments. Our analysis "
            "of 100,000 cells from breast cancer samples identifies novel cell subtypes associated with treatment resistance "
            "and potential therapeutic targets for immunotherapy approaches.",
            "Epigenetic modifications play crucial roles in developmental gene regulation. We map DNA methylation patterns "
            "across embryonic development stages, revealing dynamic changes in promoter methylation that correlate with "
            "tissue-specific gene expression programs.",
        ],
        "climate_science": [
            "Ocean circulation models predict significant weakening of the Atlantic Meridional Overturning Circulation "
            "under high-emission scenarios. Our coupled atmosphere-ocean simulations suggest a 30% reduction by 2100, "
            "with cascading effects on European climate patterns and marine ecosystems.",
            "Satellite observations combined with ground measurements reveal accelerating ice sheet mass loss in Greenland. "
            "We estimate current rates of 280 gigatons per year, contributing to 0.8mm annual sea level rise. "
            "Projections indicate potential tipping point behaviors under continued warming scenarios.",
            "Urban heat island effects exacerbate climate change impacts in megacities. Our study of 50 major cities "
            "shows nighttime temperatures 4-8 degrees higher than surrounding rural areas, with implications for "
            "energy consumption, public health, and urban planning strategies.",
            "Carbon sequestration potential of restored mangrove ecosystems exceeds that of tropical forests on a per-area "
            "basis. Field measurements across Southeast Asian restoration sites demonstrate carbon accumulation rates "
            "of 8-12 tonnes per hectare per year in sediment and biomass.",
        ],
        "public_health": [
            "Vaccine hesitancy poses a significant threat to global immunization efforts. Our survey of 25,000 adults "
            "across 15 countries identifies key drivers including misinformation exposure, institutional distrust, and "
            "lack of healthcare access. Targeted communication strategies show promise in improving uptake.",
            "Air pollution exposure during pregnancy is associated with adverse birth outcomes. Our longitudinal cohort "
            "study of 10,000 pregnancies links particulate matter exposure to increased risk of preterm birth and low "
            "birth weight, with dose-response relationships that inform regulatory standards.",
            "Mental health interventions delivered through mobile applications show efficacy comparable to traditional "
            "therapy for mild to moderate depression. Our randomized controlled trial demonstrates sustained improvements "
            "in symptom scores over 12 months with high user engagement and satisfaction.",
            "Antimicrobial resistance threatens to reverse decades of progress in infectious disease treatment. Surveillance "
            "data from hospital networks reveal rising rates of multi-drug resistant organisms, particularly in intensive "
            "care settings. Stewardship programs reduce resistance emergence by 40%.",
        ],
    }

    journals = [
        "Nature", "Science", "PNAS", "Cell", "The Lancet", "JAMA",
        "IEEE TPAMI", "NeurIPS", "Nature Genetics", "Environmental Research Letters"
    ]

    rows = []
    for i in range(200):
        field = np.random.choice(list(abstracts.keys()))
        abstract = np.random.choice(abstracts[field])
        year = np.random.choice(range(2018, 2026))
        rows.append({
            "title": f"Study {i+1}: {field.replace('_', ' ').title()} Research",
            "abstract": abstract,
            "field": field.replace("_", " ").title(),
            "year": year,
            "journal": np.random.choice(journals),
            "citations": int(np.random.exponential(scale=20)),
            "date": f"{year}-{np.random.randint(1, 13):02d}-{np.random.randint(1, 29):02d}",
        })

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df
