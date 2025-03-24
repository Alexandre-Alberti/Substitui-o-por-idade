# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 14:13:28 2025

@author: alexa
"""

import streamlit as st
import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar

# Distribuições de probabilidade
def f(x, eta, beta):
    return (beta / eta) * ((x / eta) ** (beta - 1)) * np.exp(-((x / eta) ** beta))

def F(x, eta, beta):
    return 1 - np.exp(-((x / eta) ** beta))

def R(x, eta, beta):
    return np.exp(-((x / eta) ** beta))

# Função taxa de custo
def TC(T, eta, beta, Cp, Cf, Dp, Df):
    EC = Cp * R(T, eta, beta) + Cf * F(T, eta, beta)
    EL, _ = quad(lambda x: (x + Df) * f(x, eta, beta), 0, T)
    EL += (T + Dp) * R(T, eta, beta)
    return EC / EL

# Função disponibilidade
def Disp(T, eta, beta, Dp, Df):
    ED = Dp * R(T, eta, beta) + Df * F(T, eta, beta)
    EL, _ = quad(lambda x: (x + Df) * f(x, eta, beta), 0, T)
    EL += (T + Dp) * R(T, eta, beta)
    return 1 - ED / EL

# Minimização da taxa de custo
def TC_min(eta, beta, Cp, Cf, Dp, Df):
    resultado = minimize_scalar(lambda T: TC(T, eta, beta, Cp, Cf, Dp, Df))
    return resultado.x, resultado.fun

# Maximização da disponibilidade
def Disp_max(eta, beta, Dp, Df):
    resultado = minimize_scalar(lambda T: -Disp(T, eta, beta, Dp, Df))
    return resultado.x, -resultado.fun

# Interface do Streamlit
st.title("Otimização de Política de Substituição")

# Entrada de parâmetros
eta = st.number_input("Parâmetro de Escala (η):")
beta = st.number_input("Parâmetro de Forma (β):")
Cp = st.number_input("Custo de Substituição Preventiva (Cp):")
Cf = st.number_input("Custo de Substituição Corretiva (Cf):")
Dp = st.number_input("Tempo de Parada para Substituição Preventiva (Dp):")
Df = st.number_input("Tempo de Parada para Substituição Corretiva (Df):")

# Botão para otimizar TC
if st.button("Minimizar Custo"):
    T_otm, TC_otm = TC_min(eta, beta, Cp, Cf, Dp, Df)
    st.subheader("Resultado da Otimização da Taxa de Custo")
    st.write(f"Tempo ótimo de substituição: {T_otm:.2f}")
    st.write(f"Taxa de custo mínima: {TC_otm:.4f}")

# Botão para otimizar disponibilidade
if st.button("Maximizar Disponibilidade"):
    T_otm, Disp_otm = Disp_max(eta, beta, Dp, Df)
    st.subheader("Resultado da Otimização da Disponibilidade")
    st.write(f"Tempo ótimo de substituição: {T_otm:.2f}")
    st.write(f"Disponibilidade máxima: {Disp_otm:.4f}")

# Entrada para testar uma política específica
T_teste = st.number_input("Informe um valor de T para testar a política:")
if st.button("Testar Política"):
    TC_teste = TC(T_teste, eta, beta, Cp, Cf, Dp, Df)
    Disp_teste = Disp(T_teste, eta, beta, Dp, Df)
    st.subheader("Resultados da Política Testada")
    st.write(f"Taxa de Custo: {TC_teste:.4f}")
    st.write(f"Disponibilidade: {Disp_teste:.4f}")
