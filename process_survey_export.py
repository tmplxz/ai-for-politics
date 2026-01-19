import pandas as pd


def remove_mail_address(text):
    if not isinstance(text, str):
        return text
    parts = [part if '@' not in part else '' for part in text.split()]
    return ' '.join(parts).replace('  ', ' ')


QMAP = {
    "date": "Date submitted",
    "date_s": "Date started",
    "ref": "Referrer URL",
    "age": "Please tell us your age.",
    "gender": "Please tell us your gender.",
    "ai_skill": "How do you assess your role and your own abilities with respect to AI? ",
    "qualif": "What is your highest professional qualification (according to the German Qualifications Framework)?",
    "pol_pos": "Where do you see yourself on the political spectrum? [Position]",
    "pol_part": "Which of the following best describes your current level of political participation? ",
    "Country": "In which country do you mainly reside?",
    "enc_txt": "What types of AI-generated content have you already encountered in the context of politics? [Texts]",
    "enc_img": "What types of AI-generated content have you already encountered in the context of politics? [Images]",
    "enc_vid": "What types of AI-generated content have you already encountered in the context of politics? [Videos]",
    "enc_aud": "What types of AI-generated content have you already encountered in the context of politics? [Audio]",
    "enc_med": "What types of AI-generated content have you already encountered in the context of politics? [Social Media Profiles]",
    "enc_oth": "What types of AI-generated content have you already encountered in the context of politics? [Other]",
    "usef_trnsl": "How useful do you think AI is for political education, in terms of various possible applications?  [Translation of political programs]",
    "usef_smmrz": "How useful do you think AI is for political education, in terms of various possible applications?  [Summary of political programs]",
    "usef_compr": "How useful do you think AI is for political education, in terms of various possible applications?  [Comparison of different programs]",
    "usef_vislz": "How useful do you think AI is for political education, in terms of various possible applications?  [Visualization of political programs]",
    "usef_crssm": "How useful do you think AI is for political education, in terms of various possible applications?  [Reformulation of content (e.g., for cross-media presentation)]",      
    "usef_padvc": "How useful do you think AI is for political education, in terms of various possible applications?  [Personalized advice]",
    "would_use": "Would you use AI-integrated applications for your own political education (e.g.\xa0wahl.chat\xa0oder BTW AI)? ",
    "sm1_i_use_ai": "Please rate the extent to which you agree with the following statements: [I use generative AI to form my own political opinions.]",
    "sm1_ai_unreliable": "Please rate the extent to which you agree with the following statements: [AI can analyze and visualize political content reliably and accurately.]",
    "sm1_risk_democracy": "Please rate the extent to which you agree with the following statements: [Generative AI poses a risk to democracy. ]",
    "sm1_use_respon_ai": "Please rate the extent to which you agree with the following statements: [I trust that political actors use AI responsibly.]",
    "sm1_pro_regulation": "Please rate the extent to which you agree with the following statements: [The use of AI should be more heavily regulated in the context of politics.]",
    "sm1_i_cant_recognize": "Please rate the extent to which you agree with the following statements: [I can recognize AI-generated content.]",
    "sm1_suff_discussion": "Please rate the extent to which you agree with the following statements: [The impact of AI on politics and society is being discussed sufficiently.]",
    "rel_trnsl": "We used various specialized AI models to translate the election programs from German into English, summarized them, analyzed the possible visual effects, and generated images based on the descriptions. How would you rate the reliability of generative AI for these individual tasks? [Translation of the program]",
    "rel_smmrz": "We used various specialized AI models to translate the election programs from German into English, summarized them, analyzed the possible visual effects, and generated images based on the descriptions. How would you rate the reliability of generative AI for these individual tasks? [Summary of the translation]",
    "rel_anlss": "We used various specialized AI models to translate the election programs from German into English, summarized them, analyzed the possible visual effects, and generated images based on the descriptions. How would you rate the reliability of generative AI for these individual tasks? [Analysis of the summary]",
    "rel_vislz": "We used various specialized AI models to translate the election programs from German into English, summarized them, analyzed the possible visual effects, and generated images based on the descriptions. How would you rate the reliability of generative AI for these individual tasks? [Visualization of the analyzed description]",
    "descr_1": "To what extent do the images below correspond to the following visual description?  „Increased urban greenery and tree coverage, expanded bike and pedestrian pathways, solar panel installations on buildings, de-paved and permeable surfaces, barrier-free street designs.“  []",
    "descr_2": "To what extent do the images below correspond to the following visual description?  „Increased urban greenery and tree coverage, expanded bike and pedestrian pathways, solar panel installations on buildings, de-paved and permeable surfaces, barrier-free street designs.“  [].1",
    "descr_3": "To what extent do the images below correspond to the following visual description?  „Increased urban greenery and tree coverage, expanded bike and pedestrian pathways, solar panel installations on buildings, de-paved and permeable surfaces, barrier-free street designs.“  [].2",
    "which_descr": "Which of the visual descriptions do you think best fits this image?   ",
    "same_descr": "Do you believe that these images were created using the same visual description?   ",
    "perc_changed": "Have the presented AI results changed your perception of the advantages and limitations of using GenAI in a political context?",
    "sm2_ai_for_polit": "Please indicate to what extent you agree with the following statements: [ AI technology should be used for political education]",
    "sm2_risk_democracy": "Please indicate to what extent you agree with the following statements: [AI poses a risk to democratic systems.]",
    "sm2_pro_regulation": "Please indicate to what extent you agree with the following statements: [The use of generative AI should be more heavily regulated.]",
    "sm2_public_benefit": "Please indicate to what extent you agree with the following statements: [I trust that AI is being developed for public-benefit purposes.]",
    "sm2_documentation": "Please indicate to what extent you agree with the following statements: [Models and their outputs must be documented in a more comprehensible manner.]",
    "sm2_polarization": "Please indicate to what extent you agree with the following statements: [AI promotes political polarization / reinforces differences of opinion.]",
    "qualitative": "Is there anything else you would like to share with the researchers behind this study? You are also welcome to provide your email address so that we can keep you informed about the progress of the study. ",
    "time": "Total time",
    "time_general": "Group time: General",
    "time_ai_for_politics": "Group time: AI and Political Education",
    "time_ai_feedback": "Group time: Feedback for AI-Generated Visualizations",
    "time_reflection": "Group time: Reflecting On Your Position"
}

REPLACE = {
    "I have already used them and will continue to use them in the future.": 'I have used them and will continue to do so.',
    "I haven't used them yet, but I'm considering using them in the future.": 'I have not used them, but might in the future.',
    "I have already used them, but I will not use them again.": 'I have used them, but do not want to anymore.',
    '–': '-'
}
RQMAP = {v: k for k, v in QMAP.items()}
DELIM = '$'
FNAME = 'survey_results.csv'

if __name__ == '__main__':

    df = pd.read_excel('results-survey399663.xlsx')

    # drop uncompleted responses and irrelevant columns
    df = df.loc[df['Date submitted'].dropna().index]
    df = df.drop(['Response ID', 'Last page', 'Start language', 'Date last action', 'IP address', 'Seed'] + [col for col in df.columns if col.startswith('Question time')], axis=1)
    for k, v in REPLACE.items():
        df = df.map(lambda e: e.replace(k, v) if isinstance(e, str) else e)

    # rename colums and create unique id
    df = df.rename(RQMAP, axis=1)
    df2 = df.fillna('Vxxxxx')
    df['id'] = df2.apply(lambda e: f"{e['qualif'][0]}{e['pol_pos'][0]}{e['gender'][0]}{e['ai_skill'][0]}{e['pol_part'][2]}{e['perc_changed'][0]}{str(e['age'])[:2]}{e['would_use'][2]}", axis=1)
    assert pd.unique(df['id']).size == df.shape[0]

    # reverse 1-5 scale of the first statement question for participants of the first day => afterwards aligned with other questions (higher values indicate statement support)
    first_date_responses = df[df['date'].apply(lambda e: e.startswith('2025-12-09'))]
    sm1_cols = [col for col in QMAP.keys() if col.startswith('sm1')]
    df.loc[first_date_responses.index,sm1_cols] = first_date_responses[sm1_cols] * -1 + 6

    # flip two statement questions (1-5 scale) for alignment with later statement questions
    df["sm1_ai_unreliable"] = df["sm1_ai_unreliable"] * -1 + 6
    df["sm1_i_cant_recognize"] = df["sm1_i_cant_recognize"] * -1 + 6

    # remove mail addresses from last field
    df['qualitative'] = df['qualitative'].map(remove_mail_address)

    # write to csv
    assert DELIM not in df.to_string()
    df.to_csv(FNAME, sep=DELIM, index='id')
