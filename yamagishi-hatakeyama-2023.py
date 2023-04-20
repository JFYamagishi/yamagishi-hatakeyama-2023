!pip install pulp
!pip install python-libsbml # https://pypi.org/project/python-libsbml/
!pip install simplesbml
!pip install efmtool
from pulp import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
markers = [".", ",", "o", "v", "^", "<", ">", "1", "2", "3", "4", "8", "s", "p", "*", "h", "H", "+", "x", "D"]
import random
import copy
import re
import itertools
import math
from matplotlib.ticker import MaxNLocator
import efmtool
import libsbml
import simplesbml
!pip install pulp
!pip install python-libsbml # https://pypi.org/project/python-libsbml/
!pip install simplesbml
!pip install efmtool
from pulp import *
import numpy as np
import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt
markers = [".", ",", "o", "v", "^", "<", ">", "1", "2", "3", "4", "8", "s", "p", "*", "h", "H", "+", "x", "D"]
import random
import copy
import re
import itertools
import math
from matplotlib.ticker import MaxNLocator
import efmtool
import libsbml
import simplesbml

# file_ = 'Ec_core_flux1.xml'
file_ = 'Ec_iJR904_flux1.xml'

# M_C = 0 ## in the case without $\mathcal{C}$
M_C = 1

## Set the nutrient species 
income_name = 'M_glc_D_b'

### Set stoichiometry matrix $S$ from SBML
def parse_SBML(open_file_): # Parse the SBML file
    reader = libsbml.SBMLReader()
    document = reader.readSBML(open_file_)
    sbml_model = document.getModel()

    # Get the list of all chemical species defined in the SBML model
    species = [s.getId() for s in sbml_model.getListOfSpecies()]

    # Load the SBML model using simplesbml
    model = simplesbml.loadSBMLFile(open_file_)
    boundary_species = model.getListOfBoundarySpecies()
    floating_species = model.getListOfFloatingSpecies()

    # Separate reversible and non-reversible reactions
    reactions_nonRev = []
    for reaction in sbml_model.getListOfReactions():
        reactions_nonRev.append(reaction.getId())
        if reaction.getReversible():
            reactions_nonRev.append(reaction.getId() + '_rev')

    # Separate internal and transport reactions
    reactions_int = []
    reactions_transport = []
    for reaction in sbml_model.getListOfReactions():
        if 'EX' in reaction.getId() or 't' in reaction.getId():
            reactions_transport.append(reaction.getId())
        else:
            reactions_int.append(reaction.getId())
            if reaction.getReversible():
                reactions_int.append(reaction.getId() + '_rev')
    # Create a stoichiometry matrix
    df_stoichiometry_matrix = pd.DataFrame(
        data = 0,
        index = species,
        columns = reactions_nonRev
    )
    reaction_index = 0
    for reaction in sbml_model.getListOfReactions():
        reactants = {r.getSpecies(): r.getStoichiometry() for r in reaction.getListOfReactants()}
        products = {p.getSpecies(): p.getStoichiometry() for p in reaction.getListOfProducts()}
        for species_index, species_node in enumerate(sbml_model.getListOfSpecies()):
            species_id = species_node.getId()
            net_stoichiometry = products.get(species_id, 0) - reactants.get(species_id, 0)
            df_stoichiometry_matrix.iloc[species_index, reaction_index] = net_stoichiometry
        reaction_index += 1
        if reaction.getReversible():
            for species_index, species_node in enumerate(sbml_model.getListOfSpecies()):
                species_id = species_node.getId()
                net_stoichiometry = products.get(species_id, 0) - reactants.get(species_id, 0)
                df_stoichiometry_matrix.iloc[species_index, reaction_index] = -net_stoichiometry
            reaction_index += 1
    
    metabo_names = { metabo.getId(): metabo.getName() for metabo in sbml_model.getListOfSpecies()}

    ## list up the exchangeable chemical species
    leakable = set(species)
    ### set the external conditions
    dict_intake = {k: 0 for k in leakable}
    for reaction in sbml_model.getListOfReactions():
        boundary_reactions_or_not = False
        for product in reaction.getListOfProducts():
            if "_b" in product.getSpecies(): ## for boundary reactions
                boundary_reactions_or_not = True
        if boundary_reactions_or_not:
            for parameter in reaction.getKineticLaw().getListOfParameters():
                if parameter.getId() == "FLUX_VALUE":
                    dict_intake[ product.getSpecies() ] = - parameter.getValue() 
                    if parameter.getValue() > 0.0: 
                        dict_intake[ product.getSpecies() ] = 0.0
    
    return species, boundary_species, floating_species, reactions_nonRev, reactions_int, reactions_transport, df_stoichiometry_matrix, metabo_names, leakable, dict_intake

species, Nutrients, floating_species, reactions, reactions_int, reactions_transport, S, metabo_names, leakable, dict_intake = parse_SBML(file_)

obj_reac = ''
## Set the biomass synthesis reaction as the objective reaction 
for r in reactions:
    if 'Biomass' in r:
        obj_reac = r

### Set the scale of penaly term
epsilon = 1e-6

## LP without non-stoichiometric constraint $\mathcal{C}$
def CBM_withoutC_solve(S, dict_intake, epsilon, leakable):
    m = LpProblem(sense=LpMaximize)
    S.loc['Var'] = pulp.LpVariable.dicts('v', (S.columns.values), lowBound=0, upBound=None, cat='Continuous')

    m += S.at['Var', obj_reac] - epsilon * lpDot(np.ones( len(S.index.values) ), S.loc['Var'])
    # m += S.at['Var', obj_reac] - epsilon * lpDot(np.ones( len(S.index.values) ), S.loc['Var']) + epsilon * lpDot(S.loc[income_name], S.loc['Var'])

    # Set the solution space
    for s in leakable:
        m += lpDot(S.loc[s], S.loc['Var']) + dict_intake[s] >= 0.0, s
    for k in species:
        if not (k in leakable):
            m += lpDot(S.loc[k], S.loc['Var']) == 0.0, k

    m.solve()

    return m

### Set the non-stoichiometric constraints $\mathcal{C}$
C = pd.DataFrame(np.zeros(( M_C , len(reactions) )), index=[f"R_{i+1}" for i in range(M_C)], columns=reactions)
seed = 17
random.seed( seed )
for i in range(M_C):
    for j,r in enumerate( reactions ):
        C.iloc[ i , j ] = random.uniform(0.1, 1.0)

        ## Set the costs for the forward and backward reactions as identical
        if reactions[ j - 1 ] in reactions[ j ]:
            C.iloc[ i , j - 1 ] = C.iloc[ i , j ]

## LP with non-stoichiometric constraint $\mathcal{C}$
def CBM_withC_solve(S, dict_intake, C, I_C, epsilon, leakable):
    m = LpProblem(sense=LpMaximize)
    S.loc['Var'] = pulp.LpVariable.dicts('v', (S.columns.values), lowBound=0, upBound=None, cat='Continuous')

    # Set objective function
    m += S.at['Var', obj_reac] - epsilon * lpDot( C.iloc[0], S.loc['Var'] ) - epsilon * 0.1 * lpDot(np.ones( len(S.index.values) ), S.loc['Var']) ## 2022-07-14,08-12

    # Set the solution space
    for s in leakable:
        m += lpDot(S.loc[s], S.loc['Var']) + dict_intake[s] >= 0.0, s
    for k in species:
        if not (k in leakable):
            m += lpDot(S.loc[k], S.loc['Var']) == 0.0, k
    for i in range(M_C):
        m += lpDot(C.iloc[i], S.loc['Var']) <= I_C[i], f"C{i}"

    m.solve()
    return m

### choose an effective $I_c$
I_C_max_temp = 9000.0
I_C = [ I_C_max_temp for i in range(M_C)]
I_min, dI, bins = 8.0, 0.005, 20
IList = np.arange(I_min, I_min+(bins-0.99)*dI, dI, dtype=float)
for inum, I in enumerate(IList[ : 1]):
    leakable_temp = copy.copy(leakable)
    dict_intake_temp = copy.copy(dict_intake)
    if I > 0:
        if not income_name in leakable_temp:
            leakable_temp.add( income_name )
        dict_intake_temp[ income_name ] = I

    if M_C > 0:
        m = CBM_withC_solve(S, dict_intake_temp, C, I_C, epsilon, leakable_temp)
        I_C_max_temp = value( lpDot( C.iloc[0], S.loc['Var']) )
    if M_C == 0:
        m = CBM_withoutC_solve(S, dict_intake_temp, epsilon, leakable_temp)

I_C_max = 1.0 * int( I_C_max_temp ) + 1.0
I_C = [ I_C_max for i in range(M_C)]



### Calculate metabolic responses
## Setting environmental conditions
I_min, dI, bins = 8.05, 0.0025, 2
IList = np.arange(I_min, I_min+(bins-0.99)*dI, dI, dtype=float)
dp = 0.0025
pList = np.arange(1.0, 1.0+1.01*dp, dp, dtype=float)

## Calculate the original fluxes
df_original_fluxes = pd.Series(np.zeros( len(reactions) ) , index=reactions )
dict_intake[income_name] = IList[0]
if M_C > 0:
    m = CBM_withC_solve(S, dict_intake, C, I_C, epsilon, leakable)
if M_C == 0:
    m = CBM_withoutC_solve(S, dict_intake, epsilon, leakable)
for reac_basis in reactions:
    df_original_fluxes[ reac_basis ] = value( S.at['Var', reac_basis] )

## df_leak_resps.at[ metabo , reac ] : flux of `reac` when `metabo` is additionally leaked
df_leak_resps = pd.DataFrame(np.zeros(( len(species) , len(reactions) )) , index=species , columns=reactions )
def calc_leak_response(metabo, L):
    ## Calculate the original fluxes
    df_origins_leak_c = copy.copy(df_original_fluxes)
    leakable_temp = copy.copy(leakable)
    dict_intake_temp = copy.copy(dict_intake)
    if not metabo in leakable_temp: 
        leakable_temp.add( metabo )
        dict_intake_temp[metabo] = 0.0
        if M_C > 0:
            m = CBM_withC_solve(S, dict_intake_temp, C, I_C, epsilon, leakable_temp)
        if M_C == 0:
            m = CBM_withoutC_solve(S, dict_intake_temp, epsilon, leakable_temp)
        for reac in reactions:
            df_origins_leak_c[ reac ] = value( S.at['Var', reac ] )

    ## Calculate the original fluxes at the pertuerbed condition
    leakable_temp = copy.copy(leakable)
    dict_intake_temp = copy.copy(dict_intake)
    if not metabo in leakable_temp:
        leakable_temp.add( metabo )
        dict_intake_temp[metabo] = L
    else:
        dict_intake_temp[metabo] += L
    if M_C > 0:
        m = CBM_withC_solve(S, dict_intake_temp, C, I_C, epsilon, leakable_temp)
    if M_C == 0:
        m = CBM_withoutC_solve(S, dict_intake_temp, epsilon, leakable_temp)

    ## Return the results
    df_leak_return = pd.Series(np.zeros( len(reactions) ) , index=reactions )
    for reac_i in reactions:
        df_leak_return[ reac_i ] = value( S.at['Var', reac_i] ) - df_origins_leak_c[ reac_i ]
    return df_leak_return

def calc_price_response(reac_j, p):
    df_price_resps = pd.Series(np.zeros( len(reactions) ) , index=reactions )

    ## Increase the input stoichiometry coefficients
    S_temp = copy.copy(S)
    for metabo in species:
        if S_temp.at[ metabo , reac_j] < 0:
            S_temp.at[ metabo , reac_j] *= p
    
    ## Calc. optimized reaction fluxes
    if M_C > 0:
        m = CBM_withC_solve(S_temp, dict_intake, C, I_C, epsilon, leakable)
    if M_C == 0:
        m = CBM_withoutC_solve(S_temp, dict_intake, epsilon, leakable)
    
    ## Return the results. df_price_resps[ reac_i ] : flux of `reac_i` when the price of `reac_j` is increased
    for reac_i in reactions:
        df_price_resps[ reac_i ] = value( S_temp.at['Var', reac_i] )
    return df_price_resps


## Calculate income resp.
df_leak_return = calc_leak_response(income_name, dI)
for reac in reactions:
    df_leak_resps.at[ income_name , reac ] = df_leak_return[ reac ]
## Pick up reactions with non-zero income effect
reactions_nonZeroIncomeEffect = []
for reac in reactions:
    if abs(df_leak_resps.at[ income_name , reac ]) > 0.0:
        reactions_nonZeroIncomeEffect.append( reac )

### Plot "metabolic Slutsky eq." for reactions with cotinuous price and income responses
plt.figure(figsize=(6, 6))
xy_min, xy_max = -0, 0
# for reac in reactions:
for reac in reactions_nonZeroIncomeEffect: 
    df_price_resps = calc_price_response( reac, 1.0+dp )
    
    ## Judge discontinous response or not
    df_price_resps_larger = calc_price_response( reac, 1.0 + 2.0*dp )
    price_ratio_th = 1.25
    if abs(df_price_resps[reac] - df_original_fluxes[reac]) < price_ratio_th * abs(df_price_resps_larger[reac] - df_price_resps[reac]) and abs(df_price_resps[reac] - df_original_fluxes[reac]) > abs(df_price_resps_larger[reac] - df_price_resps[reac]) / price_ratio_th : ## if the price response is continuous
        ## Calculate price change Δq^ν_i (and c^ν_μ(i)) for reaction i
        dq = 0.0
        leak_resp_Cont_or_not = True
        income_ratio_th = 1.25
        for metabo in species:
            if S.at[ metabo , reac] < 0: ## if metabo is a reactant of reac i
                if df_leak_resps.at[ metabo , reac ] == 0.0:
                    df_leak_return = calc_leak_response(metabo, dI)
                    df_leak_return_larger = calc_leak_response(metabo, 2*dI)
                    if abs(df_leak_return[reac]) > income_ratio_th * abs(df_leak_return_larger[reac] - df_leak_return[reac]) or abs(df_leak_return[reac]) < abs(df_leak_return_larger[reac] - df_leak_return[reac]) / income_ratio_th :
                        leak_resp_Cont_or_not = False
                    for reac_i in reactions:
                        df_leak_resps.at[ metabo , reac_i ] = df_leak_return[ reac_i ]
                if abs( df_leak_resps.at[ income_name , reac ] ) > 0.0:
                    dq += (-S.at[ metabo , reac]) * dp * df_leak_resps.at[ metabo , reac ] / df_leak_resps.at[ income_name , reac ]
                    # dq += (-S.at[ metabo , reac]) * dp  ## with approximation c^ν_μ ≡ 1

        if leak_resp_Cont_or_not: ## if the income and leakage responses are continuous
            responses_list = np.zeros(( len(reactions) , 2 ))
            for id_r,r in enumerate(reactions):
                rhs_and_lhs = np.array( [ 0.0, 0.0 ] )
                if not dq == 0.0:
                    rhs_and_lhs = np.array( [ round( df_original_fluxes[reac] * df_leak_resps.at[income_name , r ] / dI, 5), round( (df_price_resps[r] - df_original_fluxes[r]) / dq , 5) ] )
                responses_list[id_r, :] = rhs_and_lhs

            for id_i,i in enumerate(reactions):
                if i == reac:
                    plt.scatter(responses_list[ id_i, 0], responses_list[ id_i, 1], marker=markers[id_i-int(id_i/len(markers))*len(markers)])
                    if xy_max < responses_list[ id_i, 0]:
                        xy_max = int( responses_list[ id_i, 0] ) + 2
                    if xy_min > responses_list[ id_i, 0]:
                        xy_min = int( responses_list[ id_i, 0] ) - 2
plt.plot([xy_min , xy_max ], [-xy_min, -xy_max], label="slope -1", color="gray")
plt.grid()
plt.legend(bbox_to_anchor=(0.95, 0.95), loc='upper right', borderaxespad=0, fontsize=18)
plt.xlabel("v_i * Δv_i / ΔI_["+income_name+"]")
plt.ylabel("Δv_i / Δq_i")
plt.show()