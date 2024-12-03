import streamlit as st
import numpy as np
import pandas as pd
import app_util as util
import pickle
from sklearn.linear_model import LinearRegression
from app_util import Difficulty, Complexity
import json
import random
import altair as alt
import time
import scipy.stats as stats
import matplotlib.pyplot as plt
# Create nested dictionary with rows and columns

st.set_page_config(layout="wide")

rows = ['Low', 'Moderate', 'High']
columns = ['Easy', 'Medium', 'Hard']

questions = json.load(open("questions.json", "r"))
questions = pd.DataFrame(questions)

# pmf = util.get_starting_pmf(model)
if "complexity_pmf" not in st.session_state:
    st.session_state.complexity_pmf = {Complexity.LOW.value: .6, Complexity.MODERATE.value: .3, Complexity.HIGH.value: .1}
    st.session_state.difficulty_pmf = {Difficulty.EASY.value: .5, Difficulty.MEDIUM.value: .3, Difficulty.HARD.value: .2}

    st.session_state.complexity_beta = {Complexity.LOW.value: [3, 2], Complexity.MODERATE.value: [1, 2], Complexity.HIGH.value: [1,9]}
    st.session_state.difficulty_beta = {Difficulty.EASY.value: [1,1], Difficulty.MEDIUM.value: [1, 2], Difficulty.HARD.value: [1, 4]}

    st.session_state.answered = []

# st.title("Bayesian Updating")
# st.write(complexity_pmf)
# st.write(difficulty_pmf)

# new_complexity, new_difficulty = util.update_belief_new([3, 3], complexity_pmf, difficulty_pmf, True)
# st.write(new_complexity)
# new_complexity, new_difficulty = util.update_belief_new([3, 3], new_complexity, new_difficulty, True)
# st.write(new_complexity)

if "showed_intro" not in st.session_state:
    st.session_state.showed_intro = False

def show_welcome():
    st.title("Welcome to Kelvin's CS 109 Adaptive Testing with Probability Program!")
    st.write("This program is designed to simulate an adaptive testing environment, where the difficulty of the questions change based on your performance.")
    st.write("There is a twist: we will evaluate both your **reading comprehension scores** (complexity) and **material knowledge** scores (difficulty).")
    st.write("Research has shown that teachers often confound a student's poor reading comprehension with a lack of material knowledge.")
    st.write("To remedy that, we will present material questions that are suited to **your specific reading comprehension level.**")
    if st.button("Get Started!", type='primary'):
        st.session_state.showed_intro = True
        st.rerun()


if st.session_state.showed_intro is False:
    show_welcome()
elif st.session_state.showed_intro is True:
    left, right = st.columns(2)


    with left:
        with st.container(border=True):
            st.header("Question")
            unanswered_q = questions[~questions.index.isin(st.session_state.answered)]

            do_thompson_sampling = st.checkbox("Use Thompson Sampling", value=True)
            question = None
            if do_thompson_sampling:

                target_complexity, target_difficulty = util.thompson_sample(st.session_state.complexity_beta, st.session_state.difficulty_beta)

                try:
                    question = unanswered_q[(unanswered_q['complexity'] == target_complexity.value) & (unanswered_q['difficulty'] == target_difficulty.value)].sample(1).iloc[0]
                except Exception as e:
                    st.toast(f"Ran out of questions for {target_complexity.value} and {target_difficulty.value}!")
                    question = unanswered_q.sample(1).iloc[0]
            else:
                question = unanswered_q.sample(1).iloc[0]


            st.session_state.answered.append(question.index)
            
            st.markdown(f"**{question['question']}**")
            st.write(f"Material Difficulty: {question['difficulty']}")
            st.write(f"Reading Difficulty: {question['complexity']}")

            with st.expander("Show Answer"):
                st.markdown(f"Answer: {question['answer']}")

            new_complexity = None
            new_difficulty = None
            if st.button("I got it right!"):
                st.session_state.complexity_pmf, st.session_state.difficulty_pmf = util.update_belief_new([Complexity.from_string_to_num(question['complexity']), 
                                        Difficulty.from_string_to_num(question['difficulty'])], st.session_state.complexity_pmf, st.session_state.difficulty_pmf, True)
                st.session_state.complexity_beta[question['complexity']][0] += 1
                st.session_state.difficulty_beta[question['difficulty']][0] += 1
            if st.button("I got it wrong! (no lying 😂)"):
                st.session_state.complexity_pmf, st.session_state.difficulty_pmf = util.update_belief_new([Complexity.from_string(question['complexity']).to_num(), 
                                        Difficulty.from_string_to_num(question['difficulty'])], st.session_state.complexity_pmf, st.session_state.difficulty_pmf, False)
                st.session_state.complexity_beta[question['complexity']][1] += 1
                st.session_state.difficulty_beta[question['difficulty']][1] += 1

        util.draw_complexity_difficulty(st.session_state.complexity_pmf, st.session_state.difficulty_pmf)
                

    with right:
        util.draw_complexity_difficulty_beta(st.session_state.complexity_beta, st.session_state.difficulty_beta)
 


