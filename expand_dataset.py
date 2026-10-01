#!/usr/bin/env python3
"""Expand the seed profiles to the 50-brand, 200-creator dataset.

Generated profiles are fictional examples. Run from the repository root with
``python expand_dataset.py``. Existing seed profiles and labels are retained.
"""
import json
import random
from pathlib import Path

ROOT = Path(__file__).parent

# Brand count, industry, countries, genders, platforms, currencies, niches.
BRAND_GROUPS = [
    (5,"Fitness & Wellness",["India","US","UK","Australia","Brazil"],["female","female","all","all","male"],["instagram","youtube","tiktok"],["INR","USD","GBP","AUD","BRL"],["fitness","yoga","nutrition","wellness","sports"]),
    (4,"Beauty & Skincare",["US","UK","France","Japan"],["female","female","female","all"],["instagram","youtube","tiktok"],["USD","GBP","EUR","JPY"],["beauty","skincare","haircare","cosmetics"]),
    (3,"Gaming & Tech",["US","South Korea","Germany"],["male","male","all"],["youtube","tiktok"],["USD","KRW","EUR"],["gaming","esports","tech","streaming"]),
    (4,"Fashion & Apparel",["UK","US","Italy","India"],["female","female","male","all"],["instagram","youtube","tiktok"],["GBP","USD","EUR","INR"],["fashion","streetwear","luxury","sustainable"]),
    (3,"Fintech",["US","UK","Singapore"],["all","all","male"],["youtube","tiktok","twitter"],["USD","GBP","SGD"],["finance","tech","investing","banking"]),
    (3,"Food & Health",["US","Australia","Japan"],["all","all","female"],["instagram","youtube"],["USD","AUD","JPY"],["food","nutrition","cooking","health"]),
    (3,"Travel & Tourism",["New Zealand","India","UAE"],["all","all","female"],["instagram","youtube","tiktok"],["NZD","INR","AED"],["travel","adventure","luxury","photography"]),
    (2,"Parenting & Family",["US","Canada"],["female","all"],["instagram","youtube"],["USD","CAD"],["parenting","family","education","wellness"]),
    (3,"Consumer Tech",["US","Germany","China"],["all","all","male"],["youtube","tiktok"],["USD","EUR","CNY"],["tech","smart home","audio","mobile"]),
    (2,"EdTech",["India","US"],["all","all"],["youtube","instagram"],["INR","USD"],["education","tech","e-learning"]),
    (2,"Sustainability/Eco",["Sweden","Netherlands"],["all","all"],["instagram","youtube"],["SEK","EUR"],["sustainability","eco-friendly","lifestyle"]),
    (1,"Pet Care",["US"],["all"],["instagram","tiktok"],["USD"],["pets","animals","lifestyle"]),
    (1,"Home Decor",["Netherlands"],["female"],["instagram"],["EUR"],["home decor","interior design","lifestyle"]),
    (1,"Automotive",["Germany"],["male"],["youtube"],["EUR"],["automotive","tech","reviews"]),
    (1,"B2B SaaS",["US"],["all"],["linkedin","twitter"],["USD"],["B2B","SaaS","tech","productivity"]),
    (1,"Creator Economy",["US"],["all"],["youtube","tiktok"],["USD"],["creator tools","social media","tech"]),
    (1,"Health Tech",["Israel"],["all"],["youtube","instagram"],["ILS"],["health tech","wellness","medical"]),
]

# count, niche, countries, platform allocation, tier allocation, gender allocation
CREATOR_GROUPS = [
 (20,"fitness",["India"]*5+["US"]*4+["UK"]*2+["Brazil"]*2+["Australia"]*2+["Germany"]*2+["France","Italy","Spain"],["youtube"]*8+["instagram"]*7+["tiktok"]*5,["micro"]*6+["mid-tier"]*10+["macro"]*4,["female"]*8+["male"]*7+["all"]*5,["English","Hindi","Portuguese","Spanish","German","French","Italian"]),
 (18,"beauty",["US"]*4+["UK"]*3+["France"]*2+["Japan"]*2+["South Korea"]*2+["Italy","Spain","UAE","Canada","Australia"],["instagram"]*8+["youtube"]*6+["tiktok"]*4,["micro"]*4+["mid-tier"]*10+["macro"]*4,["female"]*14+["all"]*4,["English","French","Japanese","Korean","Spanish","Italian","Arabic"]),
 (16,"tech",["US"]*4+["UK"]*2+["Germany"]*2+["Singapore"]*2+["Israel","China","Japan","South Korea","Canada","Australia"],["youtube"]*7+["tiktok"]*5+["twitter"]*4,["micro"]*3+["mid-tier"]*9+["macro"]*4,["male"]*9+["all"]*7,["English","German","Mandarin","Japanese","Korean","Hebrew"]),
 (14,"gaming",["US"]*3+["South Korea"]*2+["Germany"]*2+["Japan"]*2+["UK","France","Brazil","Canada","Australia"],["youtube"]*6+["tiktok"]*6+["twitch"]*2,["micro"]*3+["mid-tier"]*7+["macro"]*4,["male"]*10+["all"]*4,["English","Korean","German","Japanese","Portuguese","French"]),
 (14,"food",["US"]*3+["UK"]*2+["Australia"]*2+["Japan"]*2+["Italy","Spain","India","France","Canada"],["instagram"]*6+["youtube"]*5+["tiktok"]*3,["micro"]*4+["mid-tier"]*8+["macro"]*2,["female"]*7+["all"]*7,["English","Japanese","Italian","Spanish","Hindi","French"]),
 (14,"fashion",["UK"]*3+["US"]*3+["Italy"]*2+["France"]*2+["India"]*2+["Spain","Germany"],["instagram"]*7+["youtube"]*4+["tiktok"]*3,["micro"]*4+["mid-tier"]*7+["macro"]*3,["female"]*9+["male"]*3+["all"]*2,["English","Italian","French","Hindi","Spanish","German"]),
 (10,"parenting",["US"]*3+["Canada"]*2+["UK"]*2+["Australia","Germany","Sweden"],["instagram"]*5+["youtube"]*4+["blog"],["micro"]*2+["mid-tier"]*6+["macro"]*2,["female"]*8+["all"]*2,["English","German","Swedish"]),
 (8,"luxury",["France"]*2+["Italy"]*2+["UAE"]*2+["UK","US"],["instagram"]*4+["youtube"]*3+["twitter"],["mid-tier"]*3+["macro"]*4+["mega"],["female"]*5+["all"]*3,["English","French","Italian","Arabic"]),
 (8,"travel",["New Zealand"]*2+["Thailand"]*2+["Indonesia"]*2+["Vietnam","Philippines"],["instagram"]*4+["youtube"]*3+["tiktok"],["micro"]*2+["mid-tier"]*4+["macro"]*2,["all"]*4+["female"]*4,["English","Thai","Indonesian","Vietnamese","Filipino"]),
 (10,"education",["India"]*3+["US"]*3+["UK"]*2+["Canada","Australia"],["youtube"]*5+["instagram"]*3+["tiktok"]*2,["micro"]*3+["mid-tier"]*5+["macro"]*2,["all"]*6+["female"]*4,["English","Hindi"]),
 (8,"sustainability",["Sweden"]*2+["Netherlands"]*2+["Denmark","Norway","Finland","Germany"],["instagram"]*4+["youtube"]*3+["blog"],["micro"]*3+["mid-tier"]*4+["macro"],["all"]*5+["female"]*3,["English","Swedish","Dutch","Danish","Norwegian","Finnish","German"]),
 (5,"pets",["US"]*2+["UK","Canada","Australia"],["instagram"]*3+["tiktok"]*2,["micro"]*2+["mid-tier"]*3,["all"]*3+["female"]*2,["English"]),
 (4,"home decor",["Netherlands"]*2+["US","UK"],["instagram"]*3+["youtube"],["micro"]+["mid-tier"]*3,["female"]*3+["all"],["English","Dutch"]),
 (3,"automotive",["Germany"]*2+["US"],["youtube"]*2+["instagram"],["mid-tier"]*2+["macro"],["male"]*2+["all"],["German","English"]),
 (3,"B2B/SaaS",["US"]*2+["UK"],["linkedin"]*2+["twitter"],["mid-tier"]*2+["macro"],["all"]*2+["male"],["English"]),
 (3,"creator economy",["US"]*2+["UK"],["youtube"]*2+["tiktok"],["mid-tier"]*2+["macro"],["all"]*2+["female"],["English"]),
 (2,"health tech",["Israel","US"],["youtube","instagram"],["mid-tier"]*2,["all"]*2,["Hebrew","English"]),
]

COUNTRY_CITY = {"India":"Mumbai","US":"New York","UK":"London","Australia":"Sydney","Brazil":"Sao Paulo","France":"Paris","Japan":"Tokyo","South Korea":"Seoul","Germany":"Berlin","Italy":"Milan","Spain":"Madrid","Singapore":"Singapore","Israel":"Tel Aviv","China":"Shanghai","Canada":"Toronto","UAE":"Dubai","New Zealand":"Auckland","Thailand":"Bangkok","Indonesia":"Jakarta","Vietnam":"Ho Chi Minh City","Philippines":"Manila","Sweden":"Stockholm","Netherlands":"Amsterdam","Denmark":"Copenhagen","Norway":"Oslo","Finland":"Helsinki"}
LANG = {"India":"English","Brazil":"Portuguese","France":"French","Japan":"Japanese","South Korea":"Korean","Germany":"German","Italy":"Italian","Spain":"Spanish","China":"Mandarin","Israel":"Hebrew","Thailand":"Thai","Indonesia":"Indonesian","Vietnam":"Vietnamese","Philippines":"Filipino","Sweden":"Swedish","Netherlands":"Dutch","Denmark":"Danish","Norway":"Norwegian","Finland":"Finnish"}
CURR_SYMBOL = {"INR":"₹","USD":"$","GBP":"£","AUD":"A$","BRL":"R$","EUR":"€","JPY":"¥","KRW":"₩","SGD":"S$","NZD":"NZ$","AED":"AED ","CAD":"C$","SEK":"SEK ","CNY":"CNY ","ILS":"ILS "}

def read(name):
    return json.loads((ROOT / "data" / name).read_text())

def write(name, value):
    (ROOT / "data" / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")

def add_brands():
    brands = [b for b in read("brands.json") if int(b["brand_id"][1:]) < 11]
    next_id = 11
    for count, industry, countries, genders, platforms, currencies, niches in BRAND_GROUPS:
        for i in range(count):
            country, gender, currency = countries[i], genders[i], currencies[i]
            brand_id = f"b{next_id:02}"
            budget = {"INR":500000,"USD":12000,"GBP":10000,"EUR":11000,"JPY":1500000,"AUD":16000,"BRL":60000,"KRW":15000000,"SGD":15000,"NZD":14000,"AED":45000,"CAD":15000,"SEK":120000,"CNY":80000,"ILS":40000}.get(currency,10000)
            niche = niches[i % len(niches)]
            brands.append({"brand_id":brand_id,"brand_name":f"{niche.title()} {country} {next_id}","industry":industry,"product":f"{niche.title()} products and services","campaign_title":f"{niche.title()} Creator Campaign","campaign_goal":f"Build awareness among {country} audiences interested in {niche}.","campaign_description":f"A campaign for {niche} creators in {country}.","target_audience":f"{gender.title()} audiences in {country} interested in {niche}","target_age_range":"18-40","target_gender":gender,"target_locations":[country],"required_creator_niches":[niche],"preferred_creator_niches":niches,"creator_size_preference":"mid-tier to macro","minimum_followers":50000,"maximum_followers":1000000,"budget":budget,"currency":currency,"content_types":[niche,"review","lifestyle"],"platforms":platforms,"tone":"informative and engaging","mandatory_requirements":[],"preferred_traits":["authentic","engaging"],"excluded_traits":[]})
            next_id += 1
    assert next_id == 51 and len(brands) == 50
    write("brands.json", brands)

def add_creators():
    creators = [c for c in read("creators.json") if int(c["creator_id"][1:]) < 41]
    rng = random.Random(20261001)
    names = ["Alex","Maya","Jordan","Sam","Taylor","Riley","Avery","Kai","Noah","Leah","Amara","Luca","Sofia","Ethan","Zara","Mina","Theo","Nina","Aria","Leo"]
    surnames = ["Morgan","Lee","Patel","Kim","Garcia","Brown","Chen","Singh","Martin","Wilson"]
    k = 41
    for count,niche,countries,platforms,tiers,genders,languages in CREATOR_GROUPS:
        for i in range(count):
            country, platform, tier, gender = countries[i], platforms[i], tiers[i], genders[i]
            followers = rng.randint(10000,49000) if tier == "micro" else rng.randint(50000,500000) if tier == "mid-tier" else rng.randint(500001,2000000) if tier == "macro" else rng.randint(2000001,3000000)
            lang = LANG.get(country, "English")
            if lang not in languages: lang = "English"
            audience_gender = {"female":0.72,"male":0.28} if gender == "female" else {"female":0.28,"male":0.72} if gender == "male" else {"female":0.52,"male":0.48}
            name = f"{names[(k-41)%len(names)]} {surnames[(k-41)%len(surnames)]}"
            city = COUNTRY_CITY.get(country,"Capital")
            curr = {"India":"INR","US":"USD","UK":"GBP","Australia":"AUD","Brazil":"BRL","France":"EUR","Japan":"JPY","South Korea":"KRW","Germany":"EUR","Italy":"EUR","Spain":"EUR","Singapore":"SGD","Israel":"ILS","China":"CNY","Canada":"CAD","UAE":"AED","New Zealand":"NZD","Sweden":"SEK","Netherlands":"EUR"}.get(country,"USD")
            symbol = CURR_SYMBOL[curr]
            rate_low = max(100, followers // 20)
            rate_high = rate_low * 3
            creators.append({"creator_id":f"c{k:03}","name":name,"primary_niche":niche,"secondary_niches":["lifestyle", "reviews"],"bio":f"{niche.title()} creator based in {city}, {country}. Sharing practical guides, reviews and stories for the {niche} community.","location":f"{city}, {country}","languages":[lang,"English"] if lang != "English" else ["English"],"platforms":[platform],"followers":followers,"engagement_rate":round(rng.uniform(.025,.075),3),"average_views":int(followers*rng.uniform(.2,.8)),"audience_age_range":"18-34","audience_gender_distribution":audience_gender,"audience_locations":[country],"content_types":[niche,"review","lifestyle"],"content_style":"informative and engaging","rate_card":f"{symbol}{rate_low:,}–{symbol}{rate_high:,}","rate_currency":curr,"past_brand_categories":[niche],"interests":[niche,"lifestyle"],"posting_frequency":"3-4 posts/week"})
            k += 1
    assert k == 201 and len(creators) == 200
    write("creators.json", creators)

def main():
    add_brands()
    add_creators()

if __name__ == "__main__":
    main()
