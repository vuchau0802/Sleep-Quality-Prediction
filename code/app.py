from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import json
import numpy as np
import pandas as pd
import os

app = Flask(__name__)
CORS(app) 

BASE = os.path.join(os.path.dirname(__file__), "models")
clf  = joblib.load(f"classifier.pkl")
reg  = joblib.load(f"regressor.pkl")
le   = joblib.load(f"label_encoder.pkl")
with open(f"metadata.json") as f:
    META = json.load(f)

FEATURES = META["features"]

def rec(category, priority, title, detail, tips=None):
    """Helper to build a recommendation dict."""
    r = {"category": category, "priority": priority, "title": title, "detail": detail}
    if tips:
        r["tips"] = tips
    return r
 
 
def generate_recommendations(data: dict, disorder: str, quality: float):
    recs = []
 
    # --- Unpack all inputs ---
    stress      = float(data.get("Stress_Level", 5))
    sleep_h     = float(data.get("Sleep_Duration_Hours", 7))
    caffeine    = float(data.get("Caffeine_Intake_mg", 0))
    screen      = float(data.get("Screen_Time_Before_Bed_Mins", 0))
    activity    = float(data.get("Physical_Activity_Mins", 0))
    alcohol     = float(data.get("Alcohol_Units_Per_Week", 0))
    awakenings  = float(data.get("Awakenings_Per_Night", 0))
    steps       = float(data.get("Daily_Steps", 5000))
    nap         = float(data.get("Nap_Duration_Mins", 0))
    bmi         = data.get("BMI_Category", "Normal")
    smoking     = data.get("Smoking_Status", "No")
    exercise    = data.get("Exercise_Type", "None")
    mental      = float(data.get("Mental_Health_Score", 5))
    hr          = float(data.get("Heart_Rate_BPM", 70))
    temp        = float(data.get("Room_Temperature_C", 20))
    noise       = float(data.get("Noise_Level_dB", 40))
    work_h      = float(data.get("Work_Hours_Per_Day", 8))
    age         = float(data.get("Age", 30))
    gender      = data.get("Gender", "")
    occupation  = data.get("Occupation", "")
    deficit     = float(data.get("Sleep_Deficit", max(0, 8 - sleep_h)))
    act_stress  = float(data.get("Activity_Stress_Ratio", activity / (stress + 1)))
 
    if disorder == "Insomnia":
        recs.append(rec(
            "Sleep Disorder", "Critical",
            "Insomnia Detected -- Consult a Sleep Specialist",
            "Your profile strongly matches insomnia patterns. Insomnia is treatable "
            "and you should not rely solely on sleep medication.",
            tips=[
                "Start Cognitive Behavioral Therapy for Insomnia (CBT-I) -- proven more "
                "effective than medication long-term.",
                "Practice stimulus control: use your bed only for sleep and sex, not screens or work.",
                "Use sleep restriction therapy: limit time in bed to actual sleep time, then expand gradually.",
                "Keep a sleep diary for 2 weeks before your doctor appointment.",
                "Avoid clock-watching at night -- turn clocks away from view."
            ]
        ))
    elif disorder == "Sleep Apnea":
        recs.append(rec(
            "Sleep Disorder", "Critical",
            "Sleep Apnea Risk -- Medical Evaluation Recommended",
            "Your profile suggests sleep apnea. Untreated sleep apnea raises risk of "
            "hypertension, stroke, and daytime accidents.",
            tips=[
                "Book a polysomnography (sleep study) -- it can be done at home or in a clinic.",
                "CPAP therapy resolves most cases -- modern machines are quiet and compact.",
                "Sleep on your side instead of your back to reduce airway obstruction.",
                "Avoid alcohol and sedatives before bed -- they relax throat muscles.",
                "Weight loss of even 10% can significantly reduce apnea severity."
            ]
        ))
 
    if sleep_h < 5:
        recs.append(rec(
            "Sleep Duration", "Critical",
            "Severely Insufficient Sleep -- Immediate Action Needed",
            f"You are sleeping only {sleep_h}h. Chronic sleep below 5 hours is linked to "
            "immune suppression, cognitive impairment, and cardiovascular disease.",
            tips=[
                "Add 30 minutes to your bedtime tonight -- do not wait for the weekend.",
                "Block your calendar for a fixed 8-hour sleep window and treat it as non-negotiable.",
                "Identify what is cutting your sleep (work, screens, anxiety) and address it directly.",
                "If you cannot sleep even when you have time, discuss with a doctor -- this may be insomnia.",
                "Avoid naps longer than 20 min as a substitute -- they do not replace deep sleep."
            ]
        ))
    elif sleep_h < 7:
        recs.append(rec(
            "Sleep Duration", "High",
            "Insufficient Sleep Duration",
            f"You are sleeping {sleep_h}h -- below the 7-9 hour recommendation for adults. "
            f"Your accumulated deficit is approximately {deficit:.1f}h.",
            tips=[
                "Move bedtime 15-30 minutes earlier each night until you reach 7-8 hours.",
                "Set a consistent wake time 7 days a week -- regularity matters more than duration alone.",
                "Avoid the 'I will catch up on weekends' pattern -- sleep debt does not fully repay.",
                "Track your energy and mood for one week to see how more sleep changes them.",
                "Limit late-night obligations: protect the 90 minutes before your target bedtime."
            ]
        ))
    elif sleep_h > 9.5:
        recs.append(rec(
            "Sleep Duration", "Medium",
            "Excessive Sleep -- Check for Underlying Causes",
            f"Sleeping {sleep_h}h regularly may signal poor sleep quality, depression, "
            "thyroid issues, or insufficient light exposure.",
            tips=[
                "Get bright natural light within 30 minutes of waking to regulate your circadian rhythm.",
                "Set a fixed wake time and resist staying in bed beyond it.",
                "If you still feel unrefreshed after 9+ hours, ask your doctor to check thyroid and iron levels.",
                "Evaluate depression symptoms -- hypersomnia is a common but overlooked symptom.",
                "Limit total time in bed to 8.5 hours to consolidate and improve sleep quality."
            ]
        ))
 
      if stress >= 9:
        recs.append(rec(
            "Stress", "Critical",
            "Extreme Stress -- Urgent Intervention Required",
            f"Your stress level of {int(stress)}/10 is in a dangerous range. Chronic extreme "
            "stress causes cortisol dysregulation that directly destroys sleep architecture.",
            tips=[
                "Schedule an appointment with a mental health professional this week.",
                "Identify and eliminate or delegate at least one major stressor immediately.",
                "Practice the 4-7-8 breathing technique nightly: inhale 4s, hold 7s, exhale 8s.",
                "Take a complete digital detox for at least 2 hours before bed.",
                "Consider temporary leave or reduced workload if work is the primary driver."
            ]
        ))
    elif stress >= 7:
        recs.append(rec(
            "Stress", "High",
            "High Stress Levels Disrupting Sleep",
            f"A stress level of {int(stress)}/10 is significantly elevating cortisol, "
            "making it harder to fall and stay asleep.",
            tips=[
                "Practice 10-15 minutes of mindfulness or guided meditation before bed (try Headspace or Calm).",
                "Write a 'worry list' and 'tomorrow's plan' 1 hour before bed to offload mental load.",
                "Try progressive muscle relaxation -- tense and release each muscle group from toes to face.",
                "Exercise in the morning or afternoon to metabolize stress hormones naturally.",
                "Limit news and social media after 7 PM -- information overload spikes cortisol."
            ]
        ))
    elif stress >= 5:
        recs.append(rec(
            "Stress", "Medium",
            "Moderate Stress -- Build a Pre-Sleep Wind-Down Routine",
            f"Stress at {int(stress)}/10 is manageable but worth addressing before it worsens.",
            tips=[
                "Create a 30-minute pre-sleep ritual: dim lights, herbal tea, light reading.",
                "Try box breathing (4s in, 4s hold, 4s out, 4s hold) when you feel tense.",
                "Schedule 'worry time' earlier in the day so concerns do not invade bedtime.",
                "Consider a gratitude journal -- writing 3 positives nightly reduces cortisol."
            ]
        ))
 
      if mental <= 3:
        recs.append(rec(
            "Mental Health", "Critical",
            "Very Low Mental Health Score -- Professional Support Needed",
            f"A mental health score of {int(mental)}/10 suggests significant psychological "
            "distress. Sleep and mental health have a bidirectional relationship -- each worsens the other.",
            tips=[
                "Reach out to a therapist or psychiatrist -- telehealth options are widely available.",
                "CBT-I (Cognitive Behavioral Therapy for Insomnia) treats both poor sleep and anxiety/depression simultaneously.",
                "Inform your primary care doctor -- they can screen for depression and refer appropriately.",
                "Do not use alcohol to cope with emotional distress -- it worsens both mental health and sleep.",
                "Establish one small positive routine per day (a walk, a meal with someone) to anchor your day."
            ]
        ))
    elif mental <= 5:
        recs.append(rec(
            "Mental Health", "High",
            "Low Mental Health Score Impacting Sleep Quality",
            f"Mental health at {int(mental)}/10 is a significant factor in poor sleep. "
            "Anxiety and low mood increase nighttime arousal and reduce deep sleep.",
            tips=[
                "Try mindfulness-based stress reduction (MBSR) -- an 8-week program with strong evidence.",
                "Journaling before bed can help process difficult emotions rather than ruminating.",
                "Physical exercise is one of the most effective natural mood and sleep improvers.",
                "Limit isolation -- social connection reduces anxiety and depressive symptoms.",
                "Apps like Woebot or Wysa offer CBT-based exercises between therapy sessions."
            ]
        ))
    elif mental <= 6:
        recs.append(rec(
            "Mental Health", "Medium",
            "Mental Health Could Be Strengthened",
            "Investing in mental wellbeing now prevents sleep problems from compounding.",
            tips=[
                "Practice gratitude journaling -- 5 minutes before bed noting 3 good things.",
                "Spend time in nature -- even 20 minutes outdoors reduces stress hormones.",
                "Reduce social media consumption which is linked to lower mood and worse sleep.",
                "Consider a digital mental health check-in tool like Moodfit or Daylio."
            ]
        ))
 
    if caffeine > 400:
        recs.append(rec(
            "Caffeine", "Critical",
            "Dangerously High Caffeine Intake",
            f"At {int(caffeine)}mg/day you are above the FDA safety threshold of 400mg. "
            "This level directly suppresses adenosine (your sleep pressure hormone).",
            tips=[
                "Cut intake by 100mg per week to avoid withdrawal headaches.",
                "Identify hidden caffeine sources: energy drinks, pre-workout, some teas, chocolate.",
                "Set a hard caffeine cutoff at 12 PM noon -- caffeine's half-life is 5-7 hours.",
                "Replace afternoon coffee with decaf, herbal tea, or sparkling water.",
                "Track your last caffeine each day in a notes app for one week."
            ]
        ))
    elif caffeine > 200:
        recs.append(rec(
            "Caffeine", "High",
            "High Caffeine Intake Delaying Sleep Onset",
            f"You consume {int(caffeine)}mg daily. Even afternoon caffeine can delay sleep onset "
            "by 1-2 hours and reduce deep sleep by up to 20%.",
            tips=[
                "Move your caffeine cutoff to no later than 2 PM.",
                "Reduce to 1-2 cups of coffee per day (roughly 100-200mg total).",
                "Try a 'caffeine nap': drink a small coffee then immediately nap 20 min -- wake as it kicks in.",
                "Switch to green tea (lower caffeine, contains L-theanine which promotes calm).",
                "Avoid energy drinks -- they often contain unregulated stimulant blends."
            ]
        ))
    elif caffeine > 100:
        recs.append(rec(
            "Caffeine", "Low",
            "Moderate Caffeine -- Fine-Tune Your Timing",
            f"Your {int(caffeine)}mg intake is within range but timing matters.",
            tips=[
                "Keep caffeine before 2 PM to avoid affecting sleep onset.",
                "Avoid caffeine on days when you plan to sleep earlier than usual.",
                "Notice if you need caffeine to function -- this may signal accumulated sleep debt."
            ]
        ))
 
    if alcohol > 14:
        recs.append(rec(
            "Alcohol", "Critical",
            "Very High Alcohol Consumption -- Severe Sleep Disruption",
            f"At {int(alcohol)} units/week you are consuming at a level that causes significant "
            "REM sleep suppression, rebound insomnia, and next-day fatigue.",
            tips=[
                "Consult your doctor -- at this level, abrupt cessation can cause withdrawal.",
                "Set a reduction goal of 2 fewer drinks per week until reaching safe levels.",
                "Track every drink for two weeks -- awareness alone reduces consumption.",
                "Replace evening alcohol with sparkling water, kombucha, or non-alcoholic beer.",
                "Identify your drinking triggers (stress, social situations) and address them directly."
            ]
        ))
    elif alcohol > 7:
        recs.append(rec(
            "Alcohol", "High",
            "Excess Alcohol Disrupting REM Sleep",
            f"At {int(alcohol)} units/week, alcohol is sedating you to sleep but robbing you of "
            "restorative REM cycles, causing fragmented sleep in the second half of the night.",
            tips=[
                "Target fewer than 7 units/week (UK guidelines) or 14 for men / 7 for women (US guidelines).",
                "Stop drinking at least 3 hours before bedtime.",
                "Have 2-3 alcohol-free nights per week as a starting point.",
                "Alcohol before bed increases nighttime awakenings -- track this with a sleep app.",
                "Replace the evening drink habit with a calming non-alcoholic ritual."
            ]
        ))
    elif alcohol > 3:
        recs.append(rec(
            "Alcohol", "Medium",
            "Moderate Alcohol -- Maintain Boundaries Around Sleep",
            f"{int(alcohol)} units/week is moderate -- ensure consumption does not creep up.",
            tips=[
                "Avoid drinking within 2-3 hours of bedtime even at moderate levels.",
                "Alcohol suppresses REM sleep even in small amounts -- notice your dream quality.",
                "Hydrate with one glass of water per alcoholic drink consumed."
            ]
        ))
 
    if screen > 90:
        recs.append(rec(
            "Screen Time", "High",
            "Excessive Screen Time Before Bed",
            f"{int(screen)} minutes of screens before bed significantly suppresses melatonin "
            "and delays sleep onset by shifting your circadian rhythm later.",
            tips=[
                "Set a hard screen-off alarm 60-90 minutes before your target sleep time.",
                "Enable Night Shift / f.lux / blue light filter from 6 PM onward.",
                "Replace phone time with: reading a physical book, light stretching, or a podcast.",
                "Charge your phone outside the bedroom -- removing it as a temptation.",
                "Try a 7-day screen-free bedroom challenge and note the impact on sleep."
            ]
        ))
    elif screen > 30:
        recs.append(rec(
            "Screen Time", "Medium",
            "Reduce Pre-Sleep Screen Exposure",
            f"{int(screen)} minutes of screen use before bed suppresses melatonin production.",
            tips=[
                "Aim to finish screens at least 30 minutes before bed.",
                "Use blue light blocking glasses if you cannot avoid screens in the evening.",
                "Switch to audio content (podcasts, audiobooks, sleep stories) instead of video.",
                "Set app timers on social media apps to remind you to stop."
            ]
        ))
 
    if activity == 0:
        recs.append(rec(
            "Physical Activity", "High",
            "No Physical Activity Recorded",
            "Zero daily exercise is one of the strongest predictors of poor sleep quality. "
            "Even light activity makes a significant difference.",
            tips=[
                "Start with a 10-minute walk after dinner tonight -- no gym required.",
                "Add 5 minutes per day each week until reaching 30 minutes of moderate activity.",
                "Yoga and stretching count -- even 15 minutes improves sleep onset significantly.",
                "Take the stairs, park further away, or do bodyweight exercises at home.",
                "Morning sunlight + movement resets your circadian clock powerfully."
            ]
        ))
    elif activity < 20:
        recs.append(rec(
            "Physical Activity", "High",
            "Very Low Physical Activity",
            f"Only {int(activity)} minutes/day of activity. Adults need at least 150 minutes "
            "of moderate exercise per week for healthy sleep.",
            tips=[
                "Target 20-30 minutes of brisk walking daily as a minimum baseline.",
                "Exercise in the morning or early afternoon -- avoid intense exercise within 2 hours of bed.",
                "Resistance training 2x per week significantly improves deep sleep stages.",
                "Use a fitness tracker to set a daily movement reminder.",
                "Break sedentary periods with a 5-minute walk every hour."
            ]
        ))
    elif activity < 30:
        recs.append(rec(
            "Physical Activity", "Medium",
            "Slightly Below Recommended Activity Level",
            f"At {int(activity)} minutes/day you are close to the target -- a small increase will help.",
            tips=[
                "Add 10 more minutes of walking to your daily routine.",
                "Try one new activity this week: swimming, cycling, or a fitness class.",
                "Track your resting heart rate -- improving fitness lowers it and improves sleep depth."
            ]
        ))
    elif exercise == "None" and activity >= 30:
        recs.append(rec(
            "Exercise Type", "Low",
            "Good Activity Level -- Consider Structured Exercise",
            "You are active but adding structured exercise (cardio, strength, yoga) amplifies sleep benefits.",
            tips=[
                "Yoga before bed reduces cortisol and improves sleep onset.",
                "Strength training increases slow-wave (deep) sleep -- 2x per week is enough.",
                "Cardio 3-4x per week raises body temperature, which then drops at night, promoting sleep."
            ]
        ))
 
    if steps < 3000:
        recs.append(rec(
            "Daily Movement", "High",
            "Very Low Daily Step Count",
            f"Only {int(steps)} steps/day indicates a very sedentary lifestyle, "
            "which is linked to insomnia, obesity, and cardiovascular risk.",
            tips=[
                "Set a goal of 5,000 steps first, then increase by 500/week.",
                "Walk during phone calls -- converts passive time into movement.",
                "Use a step counter app or smartwatch to make progress visible.",
                "A 20-minute lunch walk adds roughly 2,000 steps with minimal effort."
            ]
        ))
    elif steps < 6000:
        recs.append(rec(
            "Daily Movement", "Medium",
            "Below Target Step Count",
            f"At {int(steps)} steps/day you are below the 7,000-10,000 target linked "
            "to better sleep and metabolic health.",
            tips=[
                "Add a 15-minute morning or evening walk to gain ~1,500 steps.",
                "Take the stairs instead of the elevator consistently.",
                "Park 10 minutes away from your destination deliberately."
            ]
        ))
 
    if awakenings >= 5:
        recs.append(rec(
            "Sleep Continuity", "Critical",
            "Severely Fragmented Sleep",
            f"Waking {int(awakenings)} times per night severely disrupts sleep cycles. "
            "Each awakening prevents you from completing restorative deep and REM sleep stages.",
            tips=[
                "This level of fragmentation requires medical evaluation -- see a doctor.",
                "Rule out sleep apnea, restless legs syndrome, and periodic limb movement disorder.",
                "Keep a sleep log noting time of each awakening and what you were doing before bed.",
                "Avoid fluids 2 hours before bed to reduce nocturia-related waking.",
                "Ensure the bedroom is completely dark -- even small light sources trigger arousals."
            ]
        ))
    elif awakenings >= 3:
        recs.append(rec(
            "Sleep Continuity", "High",
            "Frequent Night Awakenings",
            f"Waking {int(awakenings)}x per night fragments your sleep architecture and "
            "prevents adequate deep sleep restoration.",
            tips=[
                "Check room temperature -- the ideal sleep temperature is 16-19 degrees C (60-67F).",
                "Use white noise or earplugs if ambient noise is waking you.",
                "Avoid alcohol -- it causes rebound awakenings in the second half of the night.",
                "If you wake and cannot return to sleep in 20 min, get up briefly rather than lying anxious.",
                "Consider a medical evaluation if this has persisted more than 3 months."
            ]
        ))
    elif awakenings >= 2:
        recs.append(rec(
            "Sleep Continuity", "Medium",
            "Occasional Night Awakenings -- Optimize Sleep Environment",
            "Waking once or twice is normal, but reducing it improves sleep depth.",
            tips=[
                "Cool your bedroom to 18-19 degrees C for optimal sleep.",
                "Use blackout curtains or a sleep mask to eliminate light.",
                "Try magnesium glycinate (consult a doctor) -- shown to improve sleep continuity."
            ]
        ))
 
    if hr > 90:
        recs.append(rec(
            "Heart Rate", "High",
            "Elevated Resting Heart Rate",
            f"A resting heart rate of {int(hr)} BPM is above the healthy range (60-80 BPM). "
            "High resting HR is associated with poor sleep quality and elevated stress.",
            tips=[
                "Consult your doctor to rule out cardiovascular or thyroid issues.",
                "Aerobic exercise consistently lowers resting heart rate over weeks.",
                "Practice slow diaphragmatic breathing (6 breaths/min) before bed to activate the parasympathetic system.",
                "Reduce caffeine and alcohol which both elevate heart rate.",
                "Check if any medications are raising your heart rate."
            ]
        ))
    elif hr > 80:
        recs.append(rec(
            "Heart Rate", "Medium",
            "Slightly Elevated Heart Rate",
            f"Resting HR of {int(hr)} BPM is slightly above optimal. "
            "Lower resting HR correlates with better cardiovascular fitness and deeper sleep.",
            tips=[
                "Add 20-30 minutes of cardio 3x per week to improve heart rate variability.",
                "Practice slow breathing exercises -- 5-second inhale, 5-second exhale.",
                "Reduce or eliminate caffeine after midday."
            ]
        ))
 
    if temp > 23:
        recs.append(rec(
            "Sleep Environment", "Medium",
            "Room Too Warm for Optimal Sleep",
            f"At {temp}C your bedroom is above the optimal sleep temperature range (16-19C). "
            "Core body temperature must drop ~1C to initiate and maintain sleep.",
            tips=[
                "Set your thermostat to 18-19C (65-67F) for sleeping.",
                "Use breathable cotton or bamboo bedding instead of synthetic materials.",
                "A cool shower before bed accelerates the core temperature drop.",
                "Use a fan for airflow -- it also provides white noise.",
                "If you cannot control room temperature, use lighter bedding layers."
            ]
        ))
    elif temp < 16:
        recs.append(rec(
            "Sleep Environment", "Low",
            "Room May Be Too Cold",
            f"At {temp}C your bedroom is below the comfortable sleep range. "
            "Extreme cold can cause micro-arousals and discomfort.",
            tips=[
                "Aim for 16-19C -- add a layer of bedding rather than heating the whole room.",
                "Wear warm socks to bed -- warming feet helps redistribute heat and aids sleep onset.",
                "A warm bath 1-2 hours before bed raises then drops body temperature, inducing sleepiness."
            ]
        ))
 
    if noise > 60:
        recs.append(rec(
            "Sleep Environment", "High",
            "High Noise Level Disrupting Sleep",
            f"At {int(noise)}dB your sleep environment is too noisy. "
            "Even noise that does not fully wake you causes micro-arousals and reduces deep sleep.",
            tips=[
                "Use foam earplugs (reduce noise by 25-30dB) or silicone moldable earplugs.",
                "Run a white noise machine, fan, or app (e.g. Noise Machine, myNoise) to mask variable sounds.",
                "Hang heavy curtains or add rugs to absorb sound if the noise is external traffic.",
                "If a partner's snoring is the source, address their potential sleep apnea.",
                "Consider a bedroom move or earplugs as an immediate solution."
            ]
        ))
    elif noise > 45:
        recs.append(rec(
            "Sleep Environment", "Medium",
            "Moderate Noise -- Consider Masking",
            f"At {int(noise)}dB your environment has some noise that may be disturbing sleep.",
            tips=[
                "A white noise app or fan can mask variable sounds effectively.",
                "Close windows and doors if external noise is the source.",
                "Use earplugs on nights when noise is particularly bad."
            ]
        ))
 
    if work_h >= 12:
        recs.append(rec(
            "Work-Life Balance", "High",
            "Excessive Work Hours Compressing Sleep",
            f"Working {int(work_h)} hours/day leaves insufficient time for sleep, recovery, "
            "and stress management -- all of which compound over time.",
            tips=[
                "Implement a hard work stop time -- set a calendar alarm.",
                "Log off work communication (email, Slack) at least 2 hours before bed.",
                "Discuss workload sustainability with your manager -- chronic overwork reduces productivity.",
                "Protect at least 8 hours of non-work time as a non-negotiable block.",
                "Use the first and last 30 minutes of your day as buffer -- not for reactive tasks."
            ]
        ))
    elif work_h >= 10:
        recs.append(rec(
            "Work-Life Balance", "Medium",
            "Long Work Hours -- Protect Your Wind-Down Time",
            f"At {int(work_h)} hours/day, work is likely compressing your pre-sleep recovery window.",
            tips=[
                "Create a shutdown ritual: write tomorrow's top 3 tasks, then close your laptop.",
                "Avoid checking work messages after your designated stop time.",
                "Schedule exercise and leisure like meetings -- put them in your calendar."
            ]
        ))
 
    if nap > 60:
        recs.append(rec(
            "Napping", "High",
            "Long Naps Reducing Night Sleep Pressure",
            f"Napping {int(nap)} minutes significantly reduces your sleep drive (adenosine buildup), "
            "making it harder to fall or stay asleep at night.",
            tips=[
                "Limit naps to 20 minutes maximum -- set an alarm before lying down.",
                "Avoid napping after 3 PM -- even short naps late in the day delay nighttime sleep.",
                "If you need long naps, this signals insufficient nighttime sleep -- address the root cause.",
                "Try a 'coffee nap': drink coffee immediately before a 20-min nap to wake feeling alert.",
                "If you nap for energy, prioritize adding nighttime sleep instead."
            ]
        ))
    elif nap > 30:
        recs.append(rec(
            "Napping", "Medium",
            "Naps Too Long -- Optimize Duration",
            f"A {int(nap)}-minute nap risks entering deep sleep and causing sleep inertia (grogginess) "
            "and reducing nighttime sleep pressure.",
            tips=[
                "Cap naps at 20 minutes for alertness without deep sleep inertia.",
                "Avoid naps after 3 PM -- this is the critical window for nighttime sleep drive.",
                "If you need more rest, prioritize earlier bedtime over longer napping."
            ]
        ))
 
    if bmi == "Obese":
        recs.append(rec(
            "Body Weight", "High",
            "Obesity Significantly Increases Sleep Disorder Risk",
            "Obesity is the strongest modifiable risk factor for sleep apnea and is "
            "associated with insomnia, restless legs, and poor sleep quality.",
            tips=[
                "Even 5-10% weight reduction substantially improves sleep apnea severity.",
                "Focus on reducing refined sugars and ultra-processed foods first.",
                "Start with walking 20-30 minutes daily -- sustainable low-impact exercise.",
                "Sleep deprivation itself increases hunger hormones (ghrelin) -- improving sleep aids weight loss too.",
                "Consider a referral to a dietitian or weight management clinic."
            ]
        ))
    elif bmi == "Overweight":
        recs.append(rec(
            "Body Weight", "Medium",
            "Overweight BMI -- Modest Changes Can Improve Sleep",
            "Being overweight increases airway resistance during sleep and is linked to "
            "more nighttime awakenings and lighter sleep stages.",
            tips=[
                "Target 0.5-1 kg weight loss per week through a modest calorie deficit.",
                "Increase non-exercise activity thermogenesis (NEAT) -- walking, standing, fidgeting.",
                "Avoid large meals within 3 hours of bedtime -- digestion disrupts sleep.",
                "Reduce alcohol calories -- a significant hidden source of energy intake."
            ]
        ))
 
    if smoking == "Yes":
        recs.append(rec(
            "Smoking", "High",
            "Smoking Is Directly Harming Your Sleep",
            "Nicotine is a stimulant with a 2-hour half-life. Evening smoking delays sleep "
            "onset, reduces REM sleep, and causes withdrawal micro-arousals throughout the night.",
            tips=[
                "Avoid smoking for at least 2-3 hours before your target bedtime.",
                "Nicotine replacement therapy (patch, gum) causes less sleep disruption than smoking.",
                "Quitting smoking improves sleep quality within the first 2 weeks -- a strong motivator.",
                "Seek support: call a quitline, use the NHS Stop Smoking Service, or app like Smoke Free.",
                "Tell your doctor -- they can prescribe varenicline (Champix) or bupropion which have strong quit rates."
            ]
        ))
 
    if age >= 60:
        recs.append(rec(
            "Age-Related Sleep", "Medium",
            "Age-Related Sleep Changes -- Adjust Expectations and Strategies",
            "Sleep architecture changes naturally with age: less deep sleep, earlier wake times, "
            "and more light sleep are normal -- but poor sleep is not inevitable.",
            tips=[
                "Maintain a consistent sleep-wake schedule even on weekends.",
                "Bright light therapy in the morning can help counteract advanced sleep phase.",
                "Review all medications with your doctor -- many common prescriptions disrupt sleep.",
                "Stay socially and physically active -- both are powerful sleep protectors in older adults.",
                "A short 20-minute nap is beneficial for older adults -- unlike younger people who should limit napping."
            ]
        ))
    elif age < 30:
        if sleep_h < 8:
            recs.append(rec(
                "Age-Related Sleep", "Low",
                "Young Adults Benefit from 8-9 Hours",
                "Adults under 30 still benefit from closer to 8-9 hours for full cognitive recovery.",
                tips=[
                    "Target 8 hours minimum -- your brain is still completing development processes during sleep.",
                    "Irregular sleep schedules (common in your 20s) significantly impair cognition -- stabilize yours.",
                    "Avoid all-nighters -- one night of no sleep impairs performance equivalent to being legally drunk."
                ]
            ))
 
    high_stress_jobs = ["Doctor", "Nurse", "Lawyer", "Manager", "Software Engineer"]
    shift_risk_jobs  = ["Doctor", "Nurse", "Sales Representative"]
 
    if occupation in high_stress_jobs and stress >= 6:
        recs.append(rec(
            "Occupational Health", "Medium",
            f"High-Stress Occupation ({occupation}) -- Build Deliberate Recovery",
            f"As a {occupation}, chronic occupational stress compounds sleep disruption. "
            "Building structured recovery is essential.",
            tips=[
                "Use a 'decompression commute' -- even 10 minutes of walking or audio between work and home.",
                "Set a clear physical boundary: change clothes or take a shower to signal the end of the workday.",
                "Use your lunch break to move away from your desk and eat mindfully -- not at your screen.",
                "Consider structured peer support or supervision if your work involves emotional labor."
            ]
        ))
 
    if occupation in shift_risk_jobs:
        recs.append(rec(
            "Shift Work", "Medium",
            "Shift-Work Risk -- Protect Your Sleep Schedule",
            f"Occupations like {occupation} often involve irregular hours, which disrupts the circadian rhythm.",
            tips=[
                "Use blackout curtains and a sleep mask to create darkness for daytime sleeping.",
                "Keep your sleep schedule as consistent as possible even across shift rotations.",
                "Melatonin (0.5-1mg) 30 minutes before your target sleep time can help shift your rhythm.",
                "Avoid bright light when commuting home after a night shift -- use sunglasses."
            ]
        ))
 
    if stress >= 7 and activity < 20:
        recs.append(rec(
            "Compounding Risk", "High",
            "High Stress + Low Activity -- Dangerous Combination for Sleep",
            "Stress elevates cortisol, and exercise is the most effective natural way to metabolize it. "
            "Without physical activity, stress hormones remain elevated all night.",
            tips=[
                "Even a 20-minute brisk walk daily cuts cortisol by 15-25% over time.",
                "Exercise is as effective as low-dose antidepressants for anxiety -- treat it as medicine.",
                "Yoga combines physical activity and stress reduction -- a high-leverage choice for you.",
                "Commit to one exercise session in the next 24 hours to start breaking this cycle."
            ]
        ))
 
    # High caffeine + high stress
    if caffeine > 200 and stress >= 7:
        recs.append(rec(
            "Compounding Risk", "High",
            "Caffeine + High Stress = Amplified Anxiety and Arousal",
            "Caffeine and stress both activate the sympathetic nervous system. Together they "
            "significantly worsen sleep-onset difficulties and nighttime arousal.",
            tips=[
                "Prioritize cutting caffeine as the most immediate lever -- it compounds your stress response.",
                "Switch afternoon coffee to chamomile or lemon balm tea, which have mild anxiolytic effects.",
                "Notice whether caffeine is masking fatigue caused by stress -- address the root cause."
            ]
        ))
 
    # Short sleep + high work hours
    if sleep_h < 6.5 and work_h >= 10:
        recs.append(rec(
            "Compounding Risk", "High",
            "Long Work Hours + Short Sleep = Accelerating Burnout",
            "Working 10+ hours while sleeping under 6.5 hours is a burnout trajectory. "
            "Cognitive performance drops to near-impaired levels within days.",
            tips=[
                "Counter-intuitively, sleeping more will make you more productive at work, not less.",
                "Identify which work hours are truly essential vs. habitual.",
                "Negotiate workload or deadlines before your sleep debt becomes a health crisis.",
                "Use the '2-minute rule': if something takes under 2 min, do it now to clear mental load."
            ]
        ))
 
    # Alcohol + awakenings
    if alcohol > 5 and awakenings >= 2:
        recs.append(rec(
            "Compounding Risk", "High",
            "Alcohol Is Likely Causing Your Night Awakenings",
            "Alcohol is metabolized in 3-5 hours, after which it causes a cortisol rebound "
            "that triggers awakenings -- typically in the early morning hours.",
            tips=[
                "Cut your last drink to at least 4 hours before bed and observe whether awakenings reduce.",
                "Do a 2-week alcohol-free experiment to see the direct impact on your sleep continuity.",
                "Track time of awakening -- if it is consistently 3-4 AM, alcohol metabolism is the cause."
            ]
        ))
 
    # Good profile positive reinforcement
    positives = sum([
        sleep_h >= 7 and sleep_h <= 9,
        stress <= 4,
        caffeine <= 100,
        screen <= 20,
        activity >= 45,
        alcohol <= 3,
        awakenings <= 1,
        steps >= 8000,
        mental >= 8,
        smoking == "No",
        bmi == "Normal"
    ])
    if positives >= 8:
        recs.append(rec(
            "Sleep Hygiene", "Low",
            "Excellent Sleep Hygiene -- Fine-Tune for Optimal Performance",
            "Your lifestyle habits are very strong. Focus on the marginal gains that separate "
            "good sleep from peak sleep.",
            tips=[
                "Track your Heart Rate Variability (HRV) with a wearable -- it quantifies recovery quality.",
                "Consider magnesium glycinate (200-400mg before bed) -- shown to deepen slow-wave sleep.",
                "Experiment with a consistent 90-minute pre-sleep wind-down protocol.",
                "Consider chronotype testing to align your peak work hours with your natural alertness window.",
                "Share your routine -- teaching good sleep habits to others reinforces your own."
            ]
        ))
 
    order = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
    recs.sort(key=lambda r: order.get(r["priority"], 9))
 
    # De-duplicate by title (in case combo rules overlap with individual rules)
    seen = set()
    unique_recs = []
    for r in recs:
        if r["title"] not in seen:
            seen.add(r["title"])
            unique_recs.append(r)
 
    return unique_recs  # return ALL relevant recommendations
 
 
# -- Routes
@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "models": ["classifier", "regressor"]})
 
 
@app.route("/api/metadata")
def metadata():
    return jsonify(META)
 
 
@app.route("/api/predict", methods=["POST"])
def predict():
    body = request.get_json(force=True)
 
    # Build input row
    row = {}
    for feat in FEATURES:
        val = body.get(feat)
        if val is None:
            # fill sensible defaults
            defaults = {
                "Age": 30, "Gender": "Male", "Occupation": "Engineer",
                "BMI_Category": "Normal", "Smoking_Status": "No",
                "Sleep_Duration_Hours": 7.0, "Physical_Activity_Mins": 30,
                "Stress_Level": 5, "Heart_Rate_BPM": 70, "Daily_Steps": 7000,
                "Caffeine_Intake_mg": 150, "Screen_Time_Before_Bed_Mins": 30,
                "Alcohol_Units_Per_Week": 2, "Room_Temperature_C": 20.0,
                "Noise_Level_dB": 40, "Work_Hours_Per_Day": 8,
                "Exercise_Type": "Cardio", "Mental_Health_Score": 7,
                "Awakenings_Per_Night": 1, "Nap_Duration_Mins": 0,
                "Is_Weekend": 0, "Sleep_Deficit": 0.0, "Activity_Stress_Ratio": 5.0
            }
            val = defaults.get(feat, 0)
        row[feat] = val
 
    # Recompute engineered features
    row["Sleep_Deficit"] = max(0.0, 8.0 - float(row.get("Sleep_Duration_Hours", 7)))
    row["Activity_Stress_Ratio"] = round(
        float(row.get("Physical_Activity_Mins", 30)) / (float(row.get("Stress_Level", 5)) + 1), 2
    )
 
    X = pd.DataFrame([row])[FEATURES]
 
    # Classify
    disorder_idx  = int(clf.predict(X)[0])
    disorder_prob = clf.predict_proba(X)[0].tolist()
    disorder_name = le.inverse_transform([disorder_idx])[0]
 
    # Regress
    quality_score = float(reg.predict(X)[0])
    quality_score = round(min(10, max(1, quality_score)), 2)
 
    # Recommendations
    recs = generate_recommendations(row, disorder_name, quality_score)
 
    return jsonify({
        "disorder": {
            "prediction": disorder_name,
            "probabilities": {
                le.inverse_transform([i])[0]: round(p, 4)
                for i, p in enumerate(disorder_prob)
            }
        },
        "quality_of_sleep": quality_score,
        "recommendations": recs
    })
 
 
if __name__ == "__main__":
    print("Starting Sleep Quality API on http://localhost:5050")
    app.run(port=5050, debug=False)
