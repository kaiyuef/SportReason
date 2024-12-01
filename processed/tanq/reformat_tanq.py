import json
import pandas as pd
from transformers import pipeline

import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

from utils import get_infobox

def load_jsonl(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        data = [json.loads(line) for line in file]
    return data

subset = 'dev'
filename = f'{subset}.jsonl'
data = load_jsonl(filename)
for ex in data:
    ex['seed_split'] = 'dev'
print(len(data))

subset = 'test'
filename = f'{subset}.jsonl'
data_test = load_jsonl(filename)
for ex in data_test:
    ex['seed_split'] = 'test'
data += data_test
print(len(data_test))


# class Evidence:
#     id: str  # table_1, text_1, infobox_1
#     title: str  # Title of wikipedia page
#     content: dict  # Evidence content, text or table
# text: {'text':'xxxxxx'}
# table in this format: {'columns':[a, b, c], 'rows':[[1,2,3],[1,2,3],[1,2,3]]}


bank = {'text':[],
        'infobox':[],
        'table':[]}
evidence_by_id = {}

# classifier = pipeline("text-classification", model="cardiffnlp/tweet-topic-21-multi")
classifier = pipeline("text-classification", model="wesleyacheng/news-topic-classification-with-bert")
data = data[:100]
data = [d for d in data if classifier([d['ext_question_rephrased']])[0]['label'].lower()=='sports']


for i, example in enumerate(data):
    answer_list = example['answer_list']
    example['gold_evidences']=[]
    for j in range(len(answer_list)):
        for ans in answer_list[j]["extension_answer"]:
            for pf in ans["proof"]:
                title = pf['found_in_url'].split('/')[-1]
                title = title.replace('_',' ').strip()
                pf_type = pf['proof_type']
                url = pf['found_in_url']
                meta = {'url': url}
                if pf_type=='text':
                    content = {'text': pf['text']}

                elif pf_type=='infobox':
                    infobox = get_infobox(url)
                    content = {
                        'column': infobox.columns.tolist(),
                        'rows': infobox.values.tolist()
                    }
                    meta['index'] = pf['index']
                    meta['key'] = pf['key']
                    meta['value'] = pf['value']
                    print(pf)
                    print(content,'\n\n')
                else:
                    content = pf['rows']
                    table = {'columns':[], 'rows':[]}
                    for k,row in enumerate(content):
                        row=row['cells']
                        row=[r['cell_value'] for r in row]
                        if k==0:
                            table['columns']=row
                        else:
                            table['rows'].append(row)
                    content = table
                    meta['index'] = pf['index']
                    meta['section'] = pf['section']
                    meta['parent_section'] = pf['parent_section']

                if content not in bank[pf_type]:
                    bank[pf_type].append(content)

                evidence = {
                    'id': f'{pf_type}_{len(bank[pf_type])}',
                    'title': title,
                    'content': content,
                    'meta': meta
                }

                if evidence not in example['gold_evidences']:
                    example['gold_evidences'].append(evidence)


# d: str  # sample_1
# seed_question: str # original question from source dataset
# seed_dataset: str # original dataset
# seed_id: str # original question id
# extended_question: str # our proposed question
# answer: list # correct answer, ['Biden', 'Joe Biden','President Biden']
# gold_evidence_ids: list # list of gold evidence id, ['text_1','table_2']
# gold_evidence_type: dict 
# {'text': 0/1/2…, 'table': 0/1/2…} 
# single/multiple text, single/multiple table
# gold_evidences: list # list of evidence objects
# temporal_reasoning: bool # whether involve a certain type of reasoning
# numerical_operation_program ?: str 7 - 1 = 6
# date_diff(  date_diff(November 29, 2015, March 15, 2015) , date_diff(November 26, 2017, March 26, 2017)  )
# difficulty: str low, medium, high
# meta: dict original sample info



combined_samples = []
counter = 0
for i, example in enumerate(data):
    question = example['ext_question_rephrased']
    gold_evidences = example['gold_evidences']
    gold_evidence_ids = [e['id'] for e in gold_evidences]
    sample = {
        'id': len(combined_samples),
        'seed_question': question,
        'seed_dataset': 'TANQ',
        'seed_split': example['seed_split'],
        'seed_answers': example["answer_table"],
        'seed_id': example['init_qid'],
        'extended_question': 'to be annotated',
        'answers': ['to be annotated'],
        'gold_evidence_ids': gold_evidence_ids,
        'gold_evidence_type': {
            'text': sum([eid.startswith('text') for eid in gold_evidence_ids]),
            'table': sum([eid.startswith('table') for eid in gold_evidence_ids]),
            'infobox': sum([eid.startswith('infobox') for eid in gold_evidence_ids])
        },
        'gold_evidences': gold_evidences,
        # 'temporal_reasoning': '',
        'numerical_operation_program': 'to be annotated',
        'difficulty': 'to be annotated',
        'meta': example
    }
    combined_samples.append(sample)


with open('tanq_reformatted.json', 'w') as json_file:
    json.dump(combined_samples, json_file, ensure_ascii=False)








# keywords = [
#     "Soccer (Football)", "FIFA World Cup", "UEFA Champions League", "Copa América", "English Premier League", "La Liga",
#     "Cricket", "ICC Cricket World Cup", "Indian Premier League (IPL)", "Ashes Series", "T20 World Cup", "Big Bash League (BBL)",
#     "Basketball", "NBA Finals", "FIBA World Cup", "EuroLeague", "NCAA Tournament", "Olympic Basketball Tournament",
#     "Tennis", "Wimbledon", "US Open", "French Open (Roland Garros)", "Australian Open", "ATP Finals",
#     "Golf", "The Masters", "The Open Championship", "US Open", "PGA Championship", "Ryder Cup",
#     "American Football", "Super Bowl", "NFL Playoffs", "Pro Bowl", "College Football Playoff (CFP)", "Rose Bowl",
#     "Baseball", "World Series", "MLB Playoffs", "MLB All-Star Game", "Little League World Series", "College World Series",
#     "Rugby", "Rugby World Cup", "Six Nations Championship", "Super Rugby", "Rugby Championship", "Heineken Champions Cup",
#     "Boxing", "World Boxing Championships", "The Ring Magazine Titles", "Olympic Boxing", "WBC, WBA, IBF, WBO Title Fights", "Boxing Hall of Fame Inductions",
#     "Formula 1", "Monaco Grand Prix", "Italian Grand Prix", "British Grand Prix", "Singapore Grand Prix", "Japanese Grand Prix",
#     "MMA (Mixed Martial Arts)", "UFC Championships", "Bellator MMA", "ONE Championship", "PFL Championships", "Rizin Fighting Federation",
#     "Athletics (Track and Field)", "Olympic Games", "World Athletics Championships", "Diamond League", "Commonwealth Games", "European Athletics Championships",
#     "Swimming", "Olympic Games", "FINA World Championships", "Pan Pacific Championships", "European Aquatics Championships", "World Cup Series",
#     "Cycling", "Tour de France", "Giro d'Italia", "Vuelta a España", "UCI World Championships", "Paris-Roubaix",
#     "Ice Hockey", "Stanley Cup", "IIHF World Championship", "Olympic Ice Hockey", "NHL All-Star Game", "World Junior Championship",
#     "Table Tennis", "World Table Tennis Championships", "Olympic Table Tennis", "ITTF World Tour", "World Cup", "Asian Games",
#     "Badminton", "All England Open", "BWF World Championships", "Olympic Badminton", "Thomas & Uber Cup", "Sudirman Cup",
#     "Volleyball", "FIVB Volleyball World Championship", "Olympic Volleyball", "FIVB World Cup", "Nations League", "European Championship",
#     "Racing (Car)", "Le Mans 24 Hours", "Daytona 500", "Indianapolis 500", "Formula E Championship", "World Endurance Championship",
#     "Rugby League", "Rugby League World Cup", "State of Origin", "Super League Grand Final", "NRL Grand Final", "Challenge Cup",
#     "Snooker", "World Snooker Championship", "UK Championship", "Masters Tournament", "China Open", "Champion of Champions",
#     "Darts", "PDC World Darts Championship", "Premier League Darts", "World Matchplay", "Grand Slam of Darts", "UK Open",
#     "Gymnastics", "Olympic Gymnastics", "World Artistic Gymnastics Championships", "European Championships", "Pan American Championships", "Asian Championships",
#     "Wrestling (Amateur)", "Olympic Wrestling", "World Wrestling Championships", "Pan American Wrestling Championships", "European Wrestling Championships", "Asian Wrestling Championships",
#     "Wrestling (Professional)", "WrestleMania", "Royal Rumble", "SummerSlam", "Survivor Series", "AEW Double or Nothing",
#     "Field Hockey", "FIH Hockey World Cup", "Olympic Field Hockey", "Champions Trophy", "Hockey Pro League", "European Championship",
#     "Equestrian", "Olympic Equestrian", "World Equestrian Games", "Kentucky Derby", "Royal Ascot", "FEI World Cup",
#     "Surfing", "World Surf League (WSL) Championship Tour", "Quiksilver Pro", "Rip Curl Pro", "Billabong Pipe Masters", "ISA World Surfing Games",
#     "Skiing (Alpine)", "Winter Olympics", "FIS Alpine World Ski Championships", "FIS World Cup", "Hahnenkamm", "Lauberhorn",
#     "Skiing (Nordic)", "Winter Olympics", "FIS Nordic World Ski Championships", "FIS World Cup", "Holmenkollen Ski Festival", "Vasaloppet",
#     "Snowboarding", "Winter Olympics", "X Games", "FIS Snowboarding World Championships", "US Open Snowboarding Championships", "Burton US Open",
#     "Lacrosse", "World Lacrosse Championship", "NCAA Lacrosse Championship", "Major League Lacrosse (MLL)", "Premier Lacrosse League (PLL)", "European Lacrosse Championships",
#     "Handball", "IHF World Handball Championship", "Olympic Handball", "EHF Champions League", "European Championship", "Pan American Championship",
#     "Water Polo", "FINA World Championship", "Olympic Water Polo", "World League", "European Championship", "Pan American Games",
#     "Sailing", "America's Cup", "Volvo Ocean Race", "Olympic Sailing", "World Sailing Championships", "Sydney to Hobart Yacht Race",
#     "Rowing", "Olympic Rowing", "World Rowing Championships", "Henley Royal Regatta", "Boat Race (Oxford vs Cambridge)", "European Rowing Championships",
#     "Triathlon", "Ironman World Championship", "ITU World Triathlon Series", "Olympic Triathlon", "Challenge Roth", "Escape from Alcatraz Triathlon",
#     "Martial Arts (Traditional)", "Olympic Taekwondo", "Olympic Judo", "Karate World Championships", "World Taekwondo Championships", "World Judo Championships",
#     "Skateboarding", "X Games", "Street League Skateboarding", "Vans Park Series", "Olympic Skateboarding", "Dew Tour",
#     "Horse Racing", "Kentucky Derby", "Preakness Stakes", "Belmont Stakes", "Royal Ascot", "Breeders' Cup",
#     "Motocross", "FIM Motocross World Championship", "AMA Supercross Championship", "Motocross of Nations", "Red Bull Motocross", "Monster Energy Cup",
#     "Bowling", "PBA Tour", "USBC Masters", "Bowling World Cup", "World Bowling Championships", "Weber Cup",
#     "Fencing", "Olympic Fencing", "World Fencing Championships", "European Fencing Championships", "Pan American Championships", "FIE Grand Prix",
#     "Archery", "Olympic Archery", "World Archery Championships", "World Cup", "Pan American Championships", "Asian Championships",
#     "Bobsleigh", "Winter Olympics", "World Championships", "World Cup", "European Championships", "IBSF Bobsleigh and Skeleton World Cup",
#     "Luge", "Winter Olympics", "World Championships", "World Cup", "European Championships", "FIL Luge World Cup",
#     "Skeleton", "Winter Olympics", "World Championships", "World Cup", "European Championships", "IBSF World Cup",
#     "Canoeing/Kayaking", "Olympic Canoeing/Kayaking", "World Championships", "World Cup", "European Championships", "Pan American Championships",
#     "Climbing", "Olympic Climbing", "IFSC Climbing World Championships", "World Cup", "European Championships", "Pan American Championships",
#     "Esports", "The International (Dota 2)", "League of Legends World Championship", "Overwatch League", "Call of Duty League", "Fortnite World Cup"
# ]