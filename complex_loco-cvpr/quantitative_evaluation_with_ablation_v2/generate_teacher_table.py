ids = ["Cliffs", "Stairs", "Stepping Stones", "Wave", "Obstacles"]
id_mapping = {
    "Cliffs": "cliffs",
    "Stairs": "pyramid_stairs",
    "Stepping Stones": "stepping_stons",
    "Wave": "wave",
    "Obstacles": "obstacles"
}

caption = """
\small{{\bf Evaluation of the Teacher policies.}  Place Holder}
"""

table_str = """"
\begin{table * }[t]
\caption{\small{{\bf Evaluation of the Teacher policies.}  Place Holder}}
\label{table: student_evaluation}
\centering
% \resizebox{\textwidth}{!}{begin{tabular}{lccccc}
\toprule
\multicolumn{6}{c}{Traversing Rate (\% ) $\uparrow$ }
\midrule
Scenarios & Cliffs & Stairs & Stepping Stones & Wave & Obstacles\\\midrule
Height Map Teacher & $87.7 \scriptstyle{\pm 0.5}$ & $96.0 \scriptstyle{\pm 1.2}$ & $87.4 \scriptstyle{\pm 0.5}$ & $97.2 \scriptstyle{\pm 2.9}$ & $85.1 \scriptstyle{\pm 0.7}$
% \midrule
% \multicolumn{6}{c}{Success Rate (\% ) $\uparrow$ }
% \midrule
% Height Map Teacher & $82.0 \scriptstyle{\pm 0.0}$ & $87.0 \scriptstyle{\pm 1.3}$ & $82.0 \scriptstyle{\pm 0.0}$ & $93.7 \scriptstyle{\pm 3.4}$ & $81.6 \scriptstyle{\pm 0.6}$
\bottomrule
\end{tabular}
\end{table * }
"""

print(table_str)
