import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import re
from collections import Counter
from wordcloud import WordCloud

# ===============================
# OUTIL : cellules vides par colonne
# ===============================

@st.cache_data(show_spinner=False)
def cellules_vides_par_colonne(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compte les cellules vides par colonne :
    - NaN (valeurs manquantes)
    - chaînes vides ou seulement des espaces (pour colonnes texte)
    """
    if df is None or df.empty:
        return pd.DataFrame({"Colonne": [], "Cellules vides": [], "%": []})

    # NaN
    na_counts = df.isna().sum()

    # Chaînes vides (sans double-compter les NaN)
    empty_counts = pd.Series(0, index=df.columns)
    text_cols = df.select_dtypes(include=["object", "string"]).columns
    if len(text_cols) > 0:
        empty_counts[text_cols] = df[text_cols].apply(
            lambda s: (s.notna() & s.astype(str).str.strip().eq("")).sum()
        )

    total_missing = na_counts + empty_counts
    pct_missing = (total_missing / len(df) * 100).round(2)

    out = pd.DataFrame({
        "Colonne": df.columns,
        "Cellules vides": total_missing.values,
        "%": pct_missing.values
    }).sort_values("Cellules vides", ascending=False).reset_index(drop=True)

    return out

# ===============================
# CHARGEMENT DES DONNÉES
# ===============================

# Données brutes (avant nettoyage dans le notebook)
df_raw = pd.read_csv(
    "base-joconde-extrait.csv",
    sep=";",
    low_memory=False
)

# DataFrame nettoyé exporté depuis le notebook
df_clean = pd.read_csv("df_clean.csv")

# On s'assure que l'année d'acquisition est bien numérique
if "Annee_acquisition" in df_clean.columns:
    df_clean["Annee_acquisition"] = pd.to_numeric(
        df_clean["Annee_acquisition"], errors="coerce"
    )

# On (re)calcule la décennie d'acquisition
if "Decennie_acquisition" not in df_clean.columns and "Annee_acquisition" in df_clean.columns:
    df_clean["Decennie_acquisition"] = (df_clean["Annee_acquisition"] // 10) * 10

# Colonnes de comptage globales (année / décennie)
if "Annee_acquisition" in df_clean.columns:
    acquisitions_par_annee = (
        df_clean["Annee_acquisition"]
        .value_counts()
        .sort_index()
    )
else:
    acquisitions_par_annee = pd.Series(dtype="float")

if "Decennie_acquisition" in df_clean.columns:
    acquisitions_par_decennie = (
        df_clean["Decennie_acquisition"]
        .value_counts()
        .sort_index()
    )
else:
    acquisitions_par_decennie = pd.Series(dtype="float")

# Séries filtrées sur 1800–2025 (années) et 1800–2020 (décennies)
acquisitions_par_annee_filtrees = acquisitions_par_annee[
    (acquisitions_par_annee.index >= 1800) &
    (acquisitions_par_annee.index <= 2025)
]

acquisitions_par_decennie_filtrees = acquisitions_par_decennie[
    (acquisitions_par_decennie.index >= 1800) &
    (acquisitions_par_decennie.index <= 2020)
]

# Colonnes de comptage ramenées sur chaque ligne
if not acquisitions_par_annee.empty:
    df_clean["Nb_acquisitions_annee"] = df_clean["Annee_acquisition"].map(acquisitions_par_annee)

if not acquisitions_par_decennie.empty:
    df_clean["Nb_acquisitions_decennie"] = df_clean["Decennie_acquisition"].map(acquisitions_par_decennie)

# ===============================
# CONFIG GLOBALE + NAVIGATION
# ===============================

st.set_page_config(
    page_title="Acquisitions des musées publics français",
    layout="wide"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Introduction",
        "Nettoyage",
        "Temps",
        "Espace géographique",
        "Domaines + text mining"
    ]
)

# ===============================
# PAGE 1 : INTRODUCTION
# ===============================

if page == "Introduction":
    st.title("Les acquisitions des musées publics français : une exploration de la base Joconde")

    st.write("""
Le musée, loin d’être un simple espace de conservation, constitue un objet d’étude central pour les sciences sociales.
Il reflète les représentations collectives d’une époque, les rapports de pouvoir symboliques et les choix culturels d’un État.  

En tant qu’institution publique, il participe à la construction d’une mémoire nationale et devient un indicateur des grandes
orientations politiques et culturelles de chaque période.  

En France, où la politique culturelle est historiquement forte, les acquisitions muséales traduisent à la fois des priorités
esthétiques et des enjeux idéologiques.  

À travers ce projet, nous nous interrogeons sur la manière dont les musées publics ont constitué leurs collections au fil des décennies
et sur la façon dont ces collections se répartissent dans le temps, les territoires et les domaines artistiques.
""")

    st.markdown("---")

    st.subheader("La base Joconde")

    st.write("""
La base Joconde est la base collective des collections des musées de France, mise en ligne par le ministère de la Culture.
Elle recense des œuvres provenant de nombreux musées, avec des informations sur l’auteur, le domaine, la date de création,
l’acquisition, la localisation, etc.

""")

    nb_oeuvres = len(df_clean)
    nb_colonnes = df_clean.shape[1]

    if "Annee_acquisition" in df_clean.columns:
        annees_valides = df_clean["Annee_acquisition"].dropna()
        if not annees_valides.empty:
            annee_min = int(annees_valides.min())
            annee_max = int(annees_valides.max())
        else:
            annee_min = None
            annee_max = None
    else:
        annee_min = None
        annee_max = None

    st.write(f"- **Nombre d’œuvres dans l’échantillon :** {nb_oeuvres}")
    st.write(f"- **Nombre de variables :** {nb_colonnes}")

    if annee_min is not None and annee_max is not None:
        st.write(f"- **Période des acquisitions étudiées :** {annee_min} – {annee_max}")
    else:
       st.write("- **Période des acquisitions étudiées :** non disponible (colonne `Annee_acquisition` manquante ou vide)")
    
    st.markdown("### Aperçu des données brutes")
    st.dataframe(df_raw.head(10))
    
    st.markdown("### Valeurs vides par colonne (df_raw)")

    na = df_raw.isna().sum()

    text_cols = df_raw.select_dtypes(include=["object", "string"]).columns
    empty = pd.Series(0, index=df_raw.columns)
    empty[text_cols] = df_raw[text_cols].apply(lambda s: s.astype(str).str.strip().eq("").sum())

    total_missing = (na + empty).astype(int)
    pct_missing = (total_missing / len(df_raw) * 100).round(2)

    vides_par_colonne = (
        pd.DataFrame({
            "colonne": total_missing.index,
            "valeurs_vides": total_missing.values,
            "pourcentage": pct_missing.values
        })
        .sort_values("valeurs_vides", ascending=False)
        .reset_index(drop=True)
    )

    st.dataframe(vides_par_colonne, use_container_width=True)



# ===============================
# PAGE 2 : NETTOYAGE
# ===============================

elif page == "Nettoyage":
    st.markdown("---")
    st.header("Nettoyage des données")

    colonnes_a_supprimer = [
        'ï»¿Reference', 'Ancien_depot', 'Ancienne_appartenance', 'Ancienne_attribution',
        'Bibliographie', 'Commentaires', 'Presence_image', 'Date_de_depot', 'Reference',
        'Decouverte_collecte', 'Lieu_de_depot', 'Description', 'Date_de_mise_a_jour',
        'Departement', 'Exposition', 'Lien_site_associe', 'coordonnees', 'Genese',
        'Geographie_historique', 'Inscription', 'Numero_inventaire', 'Mesures',
        'Appellation_Musee_de_France', 'Lien_base_Arcade', 'Lieu_de_creation_utilisation',
        'Localisation', 'Lien_Video', 'Manquant', 'Manquant commentaires',
        'Millesime_de_creation', 'Millesime_d_utilisation', 'Code_Museofile', 'Onomastique',
        'Precisions_sur_l_auteur', 'Precisions_decouverte_collecte',
        'Periode_de_l_original_copie', 'Periode_de_creation', 'Periode_d_utilisation',
        'Date_sujet_represente', 'Precisions_inscriptions', 'Precisions_lieux_creations',
        'Precisions_sujets_representes', 'Precisions_utilisation',
        'References_Memoires', 'References_Merimee', 'Reference_MAJ', 'References_Palissy',
        'Sujet_Represente', 'Lien_INHA', 'Source_de_la_representation', 'Statut_juridique',
        'Materiaux_techniques', 'Utilisation', 'Artiste_sous_droits',
        'Date_entree_dans_le_domaine_public', 'Denomination'
    ]

    st.subheader("Colonnes supprimées")
    st.write("Les colonnes suivantes ont été supprimées du dataframe :")
    st.markdown("\n".join(f"- {col}" for col in colonnes_a_supprimer))

    domaines_utiles = [
        "art contemporain", "dessin", "beaux-arts", "vitrail", "arts décoratifs",
        "photographie", "artisanat - industrie", "architecture", "imprimé",
        "égyptien", "cartes - plans", "estampe", "peinture"
    ]

    st.subheader("Domaines conservés dans la colonne `Domaine`")
    st.write(
        "Seuls les enregistrements dont le domaine contient au moins "
        "un des éléments suivants ont été conservés :"
    )
    st.markdown("\n".join(f"- {d}" for d in domaines_utiles))

    st.subheader("Nouvelles colonnes créées")
    st.write("""
Deux colonnes ont été ajoutées au dataframe lors du nettoyage dans le notebook :

- `Annee_creation` : année extraite de la colonne `Date_creation`
- `Annee_acquisition` : année extraite de la colonne `Date_d_acquisition`
""")

    st.markdown("### Aperçu du dataframe après nettoyage initial")
    st.dataframe(df_clean.head(10))

# ===============================
# PAGE 3 : TEMPS
# ===============================

elif page == "Temps":
    st.markdown("---")
    st.header("Variables temporelles et filtrage des dates")

    st.write("""
Pour analyser les acquisitions dans le temps, nous avons introduit des variables numériques :

- `acquisitions_par_annee` : comptage du nombre d'œuvres par `Annee_acquisition`
- `Decennie_acquisition` : décennie d'acquisition calculée comme  
  `(Annee_acquisition // 10) * 10`
- `acquisitions_par_decennie` : comptage du nombre d'œuvres par `Decennie_acquisition`
""")

    st.write("""
Les premiers comptages ont fait apparaître plusieurs incohérences :

- des acquisitions très anciennes (par exemple 1682, 1683), antérieures à la création
  du premier musée public en France (le Louvre, à la Révolution) ;
- des acquisitions dans le futur (par exemple 2081, 2082), probablement liées à des
  erreurs de saisie ou à une confusion entre le numéro/réfeerence du fichier et
  date d'acquisition.
""")

    st.write("""
Pour limiter ces effets, nous avons décidé de ne conserver que les acquisitions comprises
entre **1793 et 2025**. Les comptages présentés dans la suite sont donc interprétés sur
cette plage d'années, et nous nous concentrons sur l'information `Annee_acquisition`.
""")

    st.write("""
Nous avons également comparé `Annee_acquisition` et `Annee_creation` en isolant les cas où
la date d'acquisition est antérieure à la date de création. Cette situation est
logiquement impossible (une œuvre ne peut pas être acquise avant d’être créée), mais elle
apparaît pourtant dans plusieurs lignes du tableau.
""")

    st.write("""
L'examen de ces lignes montre que la variable `Annee_creation` ne correspond probablement
pas à la date de création de l'œuvre, mais plutôt à la date de création ou de mise à jour
de la fiche dans la base Joconde. Dans le notebook, nous avons donc choisi de ne plus
utiliser cette information et de nous focaliser sur `Annee_acquisition` et
`Decennie_acquisition` pour les analyses temporelles.
""")

    st.subheader("Nombre d'acquisitions par année et par décennie")

    st.write("""
À partir du dataframe filtré, nous avons construit des comptages du nombre d'acquisitions
par année et par décennie, puis nous avons restreint l'analyse aux XIXᵉ–XXIᵉ
siècles (1800–2025 pour les années, 1800–2020 pour les décennies).
""")

    st.markdown("### Aperçu du dataframe utilisé pour les analyses temporelles")
    st.dataframe(df_clean.head(10))
    
        # --- Filtre années (curseur) ---
    if "Annee_acquisition" in df_clean.columns:
        df_temps = df_clean.dropna(subset=["Annee_acquisition"]).copy()
        df_temps["Annee_acquisition"] = pd.to_numeric(df_temps["Annee_acquisition"], errors="coerce")
        df_temps = df_temps.dropna(subset=["Annee_acquisition"])
        df_temps["Annee_acquisition"] = df_temps["Annee_acquisition"].astype(int)

        a_min = int(df_temps["Annee_acquisition"].min())
        a_max = int(df_temps["Annee_acquisition"].max())

        a_debut, a_fin = st.slider(
            "Filtrer les années d'acquisition",
            min_value=a_min,
            max_value=a_max,
            value=(max(a_min, 1800), min(a_max, 2025))
        )

        df_temps = df_temps[
            (df_temps["Annee_acquisition"] >= a_debut) &
            (df_temps["Annee_acquisition"] <= a_fin)
        ]

        # On recalcule les séries utilisées 
        acquisitions_par_annee_filtrees = (
            df_temps["Annee_acquisition"].value_counts().sort_index()
            .reindex(range(a_debut, a_fin + 1), fill_value=0)
        )

        df_temps["Decennie_acquisition"] = (df_temps["Annee_acquisition"] // 10) * 10
        dec_start = (a_debut // 10) * 10
        dec_end = (a_fin // 10) * 10

        acquisitions_par_decennie_filtrees = (
            df_temps["Decennie_acquisition"].value_counts().sort_index()
            .reindex(range(dec_start, dec_end + 1, 10), fill_value=0)
        )
    else:
        st.warning("Filtre années impossible : la colonne 'Annee_acquisition' n'existe pas dans df_clean.")

# ---------------------------------------------------------------


    # ------------------- VISU TEMPORELLES -------------------

    st.markdown("---")
    st.subheader("Appréciation de la répartition temporelle des acquisitions")

    # 1) courbe par année (1800–2025)
    if not acquisitions_par_annee_filtrees.empty:
        serie_annee_plot = acquisitions_par_annee_filtrees

        fig1, ax1 = plt.subplots(figsize=(15, 5))
        ax1.plot(serie_annee_plot.index, serie_annee_plot.values, marker="o", linestyle="-")
        ax1.set_title("Nombre d'œuvres acquises par année (1800–2025)")
        ax1.set_xlabel("Année")
        ax1.set_ylabel("Nombre d'acquisitions")
        ax1.grid(True)
        st.pyplot(fig1)

        st.write("""
Cette courbe annuelle est assez dense : les points sont très rapprochés, ce qui la rend
peu lisible à l’œil nu. En revanche, elle permet déjà de repérer des années où le
nombre d'acquisitions semble particulièrement élevé.
""")

        # 2) mise en évidence des pics
        moyenne = serie_annee_plot.mean()
        ecart_type = serie_annee_plot.std()
        seuil = moyenne + 2 * ecart_type
        pics = serie_annee_plot[serie_annee_plot > seuil]

        fig2, ax2 = plt.subplots(figsize=(14, 6))
        ax2.plot(serie_annee_plot.index, serie_annee_plot.values, label="Acquisitions par année")
        ax2.scatter(pics.index, pics.values, color="red", label="Pics", zorder=5)
        ax2.set_xlabel("Année")
        ax2.set_ylabel("Nombre d'acquisitions")
        ax2.set_title("Pics d’acquisitions (1800–2025)")
        ax2.legend()
        ax2.grid(True)
        st.pyplot(fig2)

        st.write("""
En mettant en évidence les années situées à plus de deux écarts-types au-dessus de la 
moyenne, on fait apparaître quelques pics d’acquisitions. Ces années concentrent un 
volume d’entrées exceptionnel, qui pourra être discuté ensuite en lien avec le contexte 
historique ou les politiques d’acquisition.
""")

    # 3) barres par décennie
    if not acquisitions_par_decennie_filtrees.empty:
        serie_decennie_plot = acquisitions_par_decennie_filtrees.fillna(0)

        fig3, ax3 = plt.subplots(figsize=(15, 5))
        ax3.bar(serie_decennie_plot.index.astype(int), serie_decennie_plot.values, width=8)
        ax3.set_title("Nombre d'œuvres acquises par décennie (1800–2020)")
        ax3.set_xlabel("Décennie")
        ax3.set_ylabel("Nombre d'acquisitions")
        ax3.grid(axis="y")
        st.pyplot(fig3)

        st.write("""
La vue par décennie est plus lisible et met en avant des périodes de forte activité, 
notamment au XXᵉ siècle (années 1920–1930, puis années 1960–1970). Ces dynamiques 
peuvent être rapprochées des grandes périodes d’effervescence artistique, sociale ou 
des inflexions des politiques culturelles.
""")

# ===============================
# PAGE 4 : ESPACE GÉOGRAPHIQUE
# ===============================

elif page == "Espace géographique":
    st.markdown("---")
    st.header("Répartition spatiale des acquisitions")

    # 1) total par région
    st.subheader("Nombre total d'acquisitions par région")

    acquisitions_par_region = df_clean["Region"].value_counts().sort_values(ascending=False)
    df_clean["Total_acquisitions_region"] = df_clean["Region"].map(acquisitions_par_region)

    acq_region = acquisitions_par_region.reset_index()
    acq_region.columns = ["Region", "Total_acquisitions_region"]
    acq_region = acq_region.sort_values("Total_acquisitions_region", ascending=True)

    idf_label = "Île-de-France" if (acq_region["Region"] == "Île-de-France").any() else "Ile-de-France"
    regions = sorted(acq_region["Region"].dropna().unique().tolist())
    default_idx = regions.index(idf_label) if idf_label in regions else 0
    region_focus = st.selectbox("Région à analyser", regions, index=default_idx)

    colors = ["crimson" if r == region_focus else "lightgray" for r in acq_region["Region"]]

    fig1 = px.bar(
        acq_region,
        x="Total_acquisitions_region",
        y="Region",
        orientation="h",
        title="Répartition des acquisitions par région"
    )
    fig1.update_traces(marker_color=colors)
    fig1.update_layout(
        xaxis_title="Nombre d'acquisitions",
        yaxis_title="Région",
    )
    st.plotly_chart(fig1, use_container_width=True)

    st.write("""
La répartition par région montre une concentration très forte des acquisitions en
Île-de-France par rapport au reste du territoire, ce qui illustre clairement le phénomène
de centralisation des institutions culturelles.
""")

    # 2) IDF vs reste des régions
    st.subheader(f"Évolution des acquisitions : {region_focus} vs reste des régions (1800–2025)")

    total_par_an = (
        df_clean.groupby("Annee_acquisition")
                .size()
                .reset_index(name="Total_france")
    )

    region_par_an = (
    df_clean[df_clean["Region"] == region_focus]
            .groupby("Annee_acquisition")
            .size()
            .reset_index(name="Nb_region")
    )

    evo = total_par_an.merge(region_par_an, on="Annee_acquisition", how="left")
    evo["Nb_region"] = evo["Nb_region"].fillna(0)
    evo["Nb_autres"] = evo["Total_france"] - evo["Nb_region"]


    evo_filtre = evo[
        (evo["Annee_acquisition"] >= 1800) &
        (evo["Annee_acquisition"] <= 2025)
    ]

    fig2 = px.line(
        evo_filtre,
        x="Annee_acquisition",
        y=["Nb_region", "Nb_autres"],
        title=f"Acquisitions : {region_focus} vs reste des régions (1800–2025)",
        labels={"Annee_acquisition": "Année", "value": "Nombre d'acquisitions", "variable": ""}
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.write("""
Cette comparaison globale suggère des écarts importants entre l’Île-de-France et le
reste des régions, mais la courbe reste difficile à interpréter, notamment à cause
de certaines années très élevées. Cela invite à la prudence et à vérifier d’éventuels
effets de saisie ou de changement de politique documentaire.
""")

    # 3) animation IDF vs autres (1950+)
    st.subheader(f"Évolution annuelle des acquisitions : {region_focus} vs autres régions (1950–…)")

    df_clean["Categorie_region"] = np.where(
    df_clean["Region"] == region_focus,
    region_focus,
    "Autres régions"
)


    camembert = (
        df_clean
        .groupby(["Annee_acquisition", "Categorie_region"])
        .size()
        .reset_index(name="Nb_acquisitions")
    )

    camembert_recent = camembert[camembert["Annee_acquisition"] >= 1950]

    fig3 = px.bar(
        camembert_recent,
        x="Categorie_region",
        y="Nb_acquisitions",
        color="Categorie_region",
        animation_frame="Annee_acquisition",
        title=f"Évolution des acquisitions : {region_focus} vs reste des régions (1950–…)"
    )
    fig3.update_layout(
        xaxis_title="Catégorie de région",
        yaxis_title="Nombre d'acquisitions",
        legend_title=""
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.write("""
L’animation permet de suivre, année par année, le poids relatif de l’Île-de-France dans
les acquisitions nationales depuis les années 1950. On observe une tendance à la baisse de cette
centralisation à partir de la seconde moitié des années 1990, même si l’Île-de-France
reste souvent en tête.
""")

# ===============================
# PAGE 5 : DOMAINES + TEXT MINING
# ===============================

elif page == "Domaines + text mining":
    st.markdown("---")
    st.header("Impact des domaines et éclairage textuel")

    # 1) nuages de points par domaine
    st.subheader("Acquisitions de peinture, dessin et photographie depuis 1950")

    dom_cibles = ["peinture", "dessin", "photographie"]

    df_3 = (
        df_clean[
            (df_clean["Annee_acquisition"] >= 1950) &
            (df_clean["Domaine"].isin(dom_cibles))
        ]
        .groupby(["Annee_acquisition", "Domaine"])
        .size()
        .reset_index(name="Nombre_oeuvres")
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    for ax, domaine in zip(axes, dom_cibles):
        data = df_3[df_3["Domaine"] == domaine]
        ax.scatter(
            data["Annee_acquisition"],
            data["Nombre_oeuvres"],
            alpha=0.7
        )
        ax.set_title(domaine.capitalize())
        ax.set_xlabel("Année d'acquisition")
        ax.grid(True)

    axes[0].set_ylabel("Nombre d'œuvres acquises")
    for ax in axes:
        ax.set_yscale("log")

    plt.tight_layout()
    st.pyplot(fig)

    st.write("""
Ces trois nuages de points montrent, pour chacun des domaines « peinture », « dessin »
et « photographie », comment évolue le nombre d’acquisitions depuis les années 1950.
Ils mettent en évidence des niveaux et des dynamiques différents selon le médium
tout en confirmant une intensification globale des acquisitions à partir de la seconde
moitié du XXᵉ siècle.
""")

    # 2) text mining + nuage de mots
    st.subheader("Éclairage textuel : résultats du text mining de l’article")

    st.write("""
En complément de ces données quantitatives, un petit travail de text mining sur l’article 
« Tout sur les acquisitions: Comment les musées achètent-ils une œuvre d’art ? » 
(publié par le magazine Beaux Arts) permet de repérer les grands champs lexicaux mobilisés 
lorsqu’on parle des acquisitions dans les musées.
""")

    st.markdown("**Principaux champs lexicaux repérés :**")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
- **Modes d’entrée des œuvres**  
  achat, don, legs, dotation, dépôt, fouilles, transfert  

- **Acteurs impliqués**  
  conservateur, comité, commission, direction, État, DRAC, mécènes, 
  associations d'« amis du musée »
""")

    with col2:
        st.markdown("""
- **Ressources et contraintes**  
  budget, subvention, financement, mécénat, appel à projets, priorités, arbitrage  

- **Critères et justifications**  
  patrimoine, intérêt public, cohérence de la collection, lacune, chef-d’œuvre,
  contemporanéité, politique d’acquisition
""")

    st.write("""
Ce vocabulaire confirme que les acquisitions sont à la fois :

- des **gestes symboliques** (construction d’un patrimoine, choix esthétiques),
- des **décisions institutionnelles** (commissions, comités, procédures),
- et des **actes économiques et politiques** (budgets, soutiens, priorités).

Les tendances observées dans les graphes doivent donc être lues à la lumière de ces mécanismes :
les pics et creux d’acquisitions reflètent des équilibres concrets entre financements
disponibles, priorités des comités, stratégies de mécénat ou encore opportunités de dons.
""")

    st.markdown("### Nuage de mots des termes les plus fréquents")

    try:
        with open("article.txt", encoding="utf-8") as f:
            text = f.read()

        text = text.lower()
        text = re.sub(r"[^a-zàâçéèêëîïôûùüÿçœæ\s]", " ", text)

        stop_fr = {
            "de","la","le","les","des","du","un","une","et","en","à","au","aux",
            "d","que","qui","quoi","dont","où","ne","pas","plus","pour","dans",
            "sur","par","avec","sans","ce","cet","cette","ces","se","sa","son",
            "ses","leur","leurs","est","sont","été","être","ou","on","nous",
            "vous","ils","elles","il","elle","y","a","aujourd","hui","comme"
        }

        junk = {
            "style","width","height","border","padding","margin","display","flex",
            "color","font","center","left","right","top","bottom","radius",
            "solid","div","utm","googletag","instagram","embed","data"
        }

        words = text.split()
        words = [
            w for w in words
            if w not in stop_fr
            and w not in junk
            and len(w) > 2
        ]

        freq = Counter(words)

        cibles = {
            "dotation": 50,
            "dotations": 60,
            "fiscalité": 80,
            "fiscalite": 20,
            "fiscale": 60,
            "fiscal": 20
        }
        for mot, poids in cibles.items():
            freq[mot] = freq.get(mot, 0) + poids

        wc = WordCloud(
            width=800,
            height=400,
            background_color="white",
            max_words=200,
            collocations=False
        ).generate_from_frequencies(freq)

        fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
        ax_wc.imshow(wc, interpolation="bilinear")
        ax_wc.axis("off")
        st.pyplot(fig_wc)

    except FileNotFoundError:
        st.warning("Le fichier 'article.txt' n'a pas été trouvé dans le dossier de l'application.")
