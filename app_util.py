import streamlit as st
import numpy as np
import pandas as pd
import altair as alt    
import scipy.stats as stats
import matplotlib.pyplot as plt

# Import enum
from enum import Enum

class Difficulty(Enum):
    EASY = "Easy"
    MEDIUM = "Medium"
    HARD = "Hard"

    def to_num(self):
        if self == Difficulty.EASY:
            return 0
        elif self == Difficulty.MEDIUM:
            return 1
        else:
            return 2
    def from_num(self):
        if self == 0:
            return Difficulty.EASY
        elif self == 1:
            return Difficulty.MEDIUM
        else:
            return Difficulty.HARD
    @staticmethod
    def from_string_to_num(string):
        if string == "Easy":
            return 0
        elif string == "Medium":
            return 1
        elif string == "Hard":
            return 2
        else:
            raise ValueError(f"Invalid string, string is {string}")
class Complexity(Enum):
    LOW = "Low"
    MODERATE = "Moderate"
    HIGH = "High"

    def to_num(self):
        if self == Complexity.LOW:
            return 0
        elif self == Complexity.MODERATE:
            return 1
        else:
            return 2
    def from_num(self):
        if self == 0:
            return Complexity.LOW
        elif self == 1:
            return Complexity.MODERATE
        else:
            return Complexity.HIGH
    @staticmethod    
    def from_string(string):
        if string == "Low":
            return Complexity.LOW
        elif string == "Moderate":
            return Complexity.MODERATE
        elif string == "High":
            return Complexity.HIGH
        else:
            raise ValueError(f"Invalid string, string is {string}")
    @staticmethod
    def from_string_to_num(string):
        if string == "Low":
            return 0
        elif string == "Moderate":
            return 1
        elif string == "High":
            return 2
        else:
            raise ValueError(f"Invalid string, string is {string}")

def draw_complexity_difficulty(complexity_pmf, difficulty_pmf):
    complexity_df = pd.DataFrame(complexity_pmf.items(), columns=['Reading', 'Probability'])
    difficulty_df = pd.DataFrame(difficulty_pmf.items(), columns=['Difficulty', 'Probability'])


    complexity_chart = alt.Chart(complexity_df).mark_bar().encode(
    y=alt.Y('Reading', sort=['Low', 'Moderate', 'High']),
    x='Probability', 
    color=alt.Color('Probability', scale=alt.Scale(scheme='blues'))
    ).properties(
        width=600,
        height=200,
        title="Reading Comprehension"
    )
    difficulty_chart = alt.Chart(difficulty_df).mark_bar().encode(
    y=alt.Y('Difficulty', sort=['Easy', 'Medium', 'Hard']),
    x='Probability',
    color=alt.Color('Probability', scale=alt.Scale(scheme='greens'))
    ).properties(
        width=600,
        height=200,
        title="Material Knowledge"
    )
    st.altair_chart(complexity_chart, use_container_width=True)
    st.altair_chart(difficulty_chart, use_container_width=True)
def draw_complexity_difficulty_beta(complexity_beta, difficulty_beta):
    # Extract alpha and beta parameters
    alpha_complexity = [complexity_beta[c.value][0] for c in Complexity]
    beta_complexity = [complexity_beta[c.value][1] for c in Complexity]

    alpha_difficulty = [difficulty_beta[d.value][0] for d in Difficulty]
    beta_difficulty = [difficulty_beta[d.value][1] for d in Difficulty]

    # Generate x values
    x = np.linspace(0, 1, 100)

    # Calculate PDFs
    pdf_complexity = [stats.beta.pdf(x, a, b) for a, b in zip(alpha_complexity, beta_complexity)]
    pdf_difficulty = [stats.beta.pdf(x, a, b) for a, b in zip(alpha_difficulty, beta_difficulty)]

    # Function to create an Altair chart
    def create_altair_chart(x, pdf_values, title):
        data = pd.DataFrame({'x': x, 'pdf': pdf_values})
        
        chart = alt.Chart(data).mark_line().encode(
            x=alt.X('x', title='x'),
            y=alt.Y('pdf', title='Probability Density'),
            tooltip=['x', 'pdf']
        ).properties(
            title=title,
            width=200,
            height=300
        ).interactive()
        
        return chart

    st.title("Beta Distributions for Complexity and Difficulty")

    # Create a 3x3 grid of charts
    cols_complexity = st.columns(3)
    for i, (c_name, pdf) in enumerate(zip(Complexity, pdf_complexity)):
        with cols_complexity[i]:
            chart = create_altair_chart(x, pdf, f'Reading Difficulty: {c_name.value}')
            st.altair_chart(chart, use_container_width=True)

    cols_difficulty = st.columns(3)
    for i, (d_name, pdf) in enumerate(zip(Difficulty, pdf_difficulty)):
        with cols_difficulty[i]:
            chart = create_altair_chart(x, pdf, f'Material Difficulty: {d_name.value}')
            st.altair_chart(chart, use_container_width=True)

def thompson_sample(complexity_beta, difficulty_beta):
    complexity_samples = [np.random.beta(a, b) for a, b in complexity_beta.values()]
    difficulty_samples = [np.random.beta(a, b) for a, b in difficulty_beta.values()]

    target_complexity = Complexity.from_num(np.argmax(complexity_samples))
    target_difficulty = Difficulty.from_num(np.argmax(difficulty_samples))

    return target_complexity, target_difficulty

def get_starting_pmf(model):
    rows = ['Low', 'Moderate', 'High']
    columns = ['Easy', 'Medium', 'Hard']
    prior = {row: {col: 1/9 for col in columns} for row in rows}
    total = 0
    for c in Complexity:
        for d in Difficulty:
            # st.write("C: ", c.value)
            # st.write("D: ", d.value)
            sub_normalization = model.predict(np.array([c.to_num(), d.to_num()]).reshape(1, -1))[0] / 100 * prior[c.value][d.value]
            # st.write(f"Sub normalization: {sub_normalization}")
            prior[c.value][d.value] = sub_normalization
            total += sub_normalization
    for c in Complexity:
        for d in Difficulty:
            prior[c.value][d.value] /= total
    return prior
def update_belief_new(question_data : np.array, complexity_pmf, difficulty_pmf, is_correct):
    question_complexity = Complexity.from_num(question_data[0])
    question_difficulty = Difficulty.from_num(question_data[1])

    complexity_sum = 0
    for c in Complexity:
        # 70 percent chance of getting the question right if the user is at the same complexity level
        likelihood = 0
        if c == question_complexity:
            likelihood = complexity_pmf[c.value] * .65
        elif c == Complexity.LOW and question_complexity == Complexity.MODERATE:
            likelihood = complexity_pmf[c.value] * .4
        elif c == Complexity.LOW and question_complexity == Complexity.HIGH:
            likelihood = complexity_pmf[c.value] * .1
        elif c == Complexity.MODERATE and question_complexity == Complexity.LOW:
            likelihood = complexity_pmf[c.value] * .8
        elif c == Complexity.MODERATE and question_complexity == Complexity.HIGH:
            likelihood = complexity_pmf[c.value] * .2
        elif c == Complexity.HIGH and question_complexity == Complexity.LOW:
            likelihood = complexity_pmf[c.value] * .9
        elif c == Complexity.HIGH and question_complexity == Complexity.MODERATE:
            likelihood = complexity_pmf[c.value] * .7
        if not is_correct:
            likelihood = 1 - likelihood
        complexity_pmf[c.value] = likelihood
        complexity_sum += complexity_pmf[c.value]
    for c in Complexity:
        complexity_pmf[c.value] /= complexity_sum

    # do the same for difficulty
    difficulty_sum = 0
    for d in Difficulty:
        difficulty_likelihood = 0
        # 70 percent chance of getting the question right if the user is at the same difficulty level
        if d == question_difficulty:
            difficulty_likelihood = difficulty_pmf[d.value] * .65
        elif d == Difficulty.EASY and question_difficulty == Difficulty.MEDIUM:
            difficulty_likelihood = difficulty_pmf[d.value] * .4
        elif d == Difficulty.EASY and question_difficulty == Difficulty.HARD:
            difficulty_likelihood = difficulty_pmf[d.value] * .1
        elif d == Difficulty.MEDIUM and question_difficulty == Difficulty.EASY:
            difficulty_likelihood = difficulty_pmf[d.value] * .8
        elif d == Difficulty.MEDIUM and question_difficulty == Difficulty.HARD:
            difficulty_likelihood = difficulty_pmf[d.value] * .2
        elif d == Difficulty.HARD and question_difficulty == Difficulty.EASY:
            difficulty_likelihood = difficulty_pmf[d.value] * .9
        elif d == Difficulty.HARD and question_difficulty == Difficulty.MEDIUM:
            difficulty_likelihood = difficulty_pmf[d.value] * .7
        if not is_correct:
            difficulty_likelihood = 1 - difficulty_likelihood
        difficulty_pmf[d.value] = difficulty_likelihood
        difficulty_sum += difficulty_pmf[d.value]

    for d in Difficulty:
        difficulty_pmf[d.value] /= difficulty_sum
    
    return complexity_pmf, difficulty_pmf
def update_belief(model, prior : dict, is_correct : bool, data : np.array):
    """
    Update the prior belief with new data.
    
    """
    posterior = prior.copy()
    p_correctness = 0
    likelihood = model.predict(data.reshape(1, -1))[0] / 100
    if not is_correct:
        likelihood = 1 - likelihood
    prior_val = prior[Complexity.from_num(data[0]).value][Difficulty.from_num(data[1]).value]

    posterior[Complexity.from_num(data[0]).value][Difficulty.from_num(data[1]).value] = likelihood * prior_val
    # for c in Complexity:
    #     for d in Difficulty:
    #         posterior[c.value][d.value] = model.predict(np.array([c.to_num(), d.to_num()]).reshape(1, -1))[0] / 100 * prior[c.value][d.value]
    
    return posterior
    
    # likelihood = model.predict(data.reshape(1, -1))[0] / 100
    # if not is_correct:
    #     likelihood = 1 - likelihood
    # st.write(f"LIKELIHOOD: {likelihood}")
    # complexity = Complexity.from_num(data[0])
    # difficulty = Difficulty.from_num(data[1])

    # prior_val = prior[complexity.value][difficulty.value]
    # st.write(f"PRIOR: {prior_val}")

    # normalization = 0
    # for c in Complexity:
    #     for d in Difficulty:
    #         # st.write("C: ", c.value)
    #         # st.write("D: ", d.value)
    #         sub_normalization = model.predict(np.array([c.to_num(), d.to_num()]).reshape(1, -1))[0] / 100 * prior[c.value][d.value]
    #         # st.write(f"Sub normalization: {sub_normalization}")
    #         normalization += sub_normalization
    # # st.write(f"Normalization: {normalization}")
    # prior[complexity.value][difficulty.value] = likelihood * prior_val / normalization

    return prior
def normalize(pmf):
    total = 0
    for c in Complexity:
        for d in Difficulty:
            total += pmf[c.value][d.value]
    for c in Complexity:
        for d in Difficulty:
            pmf[c.value][d.value] /= total
    return pmf