# Delamination beam solver

- beam_solver.py: Python/NumPy solver (global-local quasi-2D beam element, bilinear law, one-parameter damage).
- score_solver.py: runs it on three beams and prints the error and the score.
- closed_form_check.py: closed-form laminated beam check.
- simple_answers.py: simple answers and bilinear-law check.

Requires Python 3.12 and NumPy. The last three scripts read stored results from data/, which is not included here.

The formulation is the author's. The Python code is a rewrite of part of the author's MATLAB software. Mode "as_coded" keeps the conventions of that software (rotated-stiffness terms b16 and b36, and one interface
factor taken from plies 1 and 2); mode "textbook" uses the standard expressions and does not reproduce its opening.
