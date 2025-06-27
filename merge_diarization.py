import pandas as pd
import io
import os

def fusionner_segments(df, seuil_de_fusion_sec):
    """
    Fusionne les segments de parole d'un même locuteur s'ils sont séparés 
    par un silence inférieur au seuil.

    Args:
        df (pd.DataFrame): DataFrame contenant la diarisation avec les colonnes 
                           ['start', 'end', 'speaker'].
        seuil_de_fusion_sec (float): Le seuil en secondes. Si l'écart entre 
                                     deux segments du même locuteur est inférieur 
                                     ou égal à ce seuil, ils sont fusionnés.

    Returns:
        pd.DataFrame: Un nouveau DataFrame avec les segments fusionnés.
    """
    if df.empty:
        return df

    # S'assurer que le dataframe est trié par temps de début
    df = df.sort_values(by='start').reset_index(drop=True)

    segments_fusionnes = []
    # Initialiser avec le premier segment
    segment_en_cours = df.iloc[0].to_dict()

    for i in range(1, len(df)):
        segment_suivant = df.iloc[i].to_dict()
        
        # Calcul de l'écart (gap) entre la fin du segment en cours et le début du suivant
        ecart = segment_suivant['start'] - segment_en_cours['end']

        # Condition de fusion : même locuteur et écart inférieur au seuil
        if (segment_en_cours['speaker'] == segment_suivant['speaker'] and 
            ecart <= seuil_de_fusion_sec):
            # On fusionne en mettant à jour la fin du segment en cours
            segment_en_cours['end'] = segment_suivant['end']
        else:
            # Sinon, on ajoute le segment en cours à notre liste et on passe au suivant
            segments_fusionnes.append(segment_en_cours)
            segment_en_cours = segment_suivant
            
    # Ne pas oublier d'ajouter le tout dernier segment
    segments_fusionnes.append(segment_en_cours)

    return pd.DataFrame(segments_fusionnes)

# --- Exemple d'utilisation ---

# 1. Vos données 
# Pour un vrai fichier, utilisez : df = pd.read_csv('votre_fichier.csv')
SEUIL_FUSION_SECONDES = 1.5

for i in range(1, 11):
    input_path = f"V0DataSet/Diarization_Results/{i}_video_diarization_reclustered.csv"
    output_path = f"V0DataSet/Diarization_Results/{i}_video_diarization_fusionnee.csv"
    if not os.path.exists(input_path):
        print(f"Fichier non trouvé : {input_path}")
        continue

    df = pd.read_csv(input_path)
    df_fusionne = fusionner_segments(df, SEUIL_FUSION_SECONDES)

    print(f"\n--- {input_path} ---")
    print(df_fusionne)

    df_fusionne.to_csv(output_path, index=False)
