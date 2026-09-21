# Carga de librerías 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

# Carga de datos 
df = pd.read_csv("smart_workout_raw_dataset.csv") 
# Dimensiones del DataFrame 
df.shape 
# Impresión de las primeras filas 
df.head() 
# Identificación del tipo de cada variable 
df.info() 
# Creando nuevas variables con valores en minúsculas y sin espacios vacíos 
categorical_columns = [ 
"persona", 
"fitness_goal", 
"experience_level", 
"injury_constraint", 
"bodyPart", 
"equipment", 
"target_muscle" 
] 
for column in categorical_columns: 
    df[column + "_std"] = ( 
    df[column] 
    .astype("string") 
    .str.strip() 
    .str.lower() 
    .str.replace(" ", "_") 
    ) 
df[["fitness_goal", "fitness_goal_std"]].drop_duplicates().sort_values("fitness_goal_std") 
df_usuario = ( df.groupby("user_id") .agg( 
    age = ("age", "median"), 
    n_records = ("user_id", "size"), 
    n_exercises = ("exercise_id", "nunique"), 
    n_bodyparts = ("bodyPart_std", "nunique"), 
    n_equipment = ("equipment_std", "nunique"), 
    n_target_muscles = ("target_muscle_std", "nunique"), 
    mean_rating = ("rating", "mean"), 
    median_rating = ("rating", "median"), 
    sd_rating = ("rating", "std") ) 
.reset_index() 
) 
# Dimensiones del nuevo DataFrame 
df_usuario.shape 
# Impresión de las primeras filas 
df_usuario.head() 
# Calculando el conteo de objetivos por usuario 
conteo_objetivos = pd.crosstab( 
df["user_id"], 
df["fitness_goal_std"] 
) 
# Añadiendo al nombre de cada variable el prefijo n_goal_ 
conteo_objetivos = conteo_objetivos.add_prefix("n_goal_") 
# Calculando el conteo de partes del cuerpo ejercitadas por usuario 
conteo_bodyparts = pd.crosstab( 
    df["user_id"], 
    df["bodyPart_std"] 
) 
# Calculando el conteo de partes del cuerpo ejercitadas por usuario 
conteo_bodyparts = pd.crosstab( 
    df["user_id"], 
    df["bodyPart_std"] 
) 
# Agregando variables que cuentan objetivos 
df_usuario = df_usuario.merge( 
    conteo_objetivos, 
    on = "user_id", 
    how = "left" 
) 
# Agregando variables que cuentan partes del cuerpo ejercitadas 
df_usuario = df_usuario.merge( 
    conteo_bodyparts, 
    on="user_id", 
    how="left" 
) 
# Proporción de cantidad de registros en general fitness 
df_usuario["share_general_fitness"] = ( 
    df_usuario["n_goal_general_fitness"] 
    / df_usuario["n_records"] 
) 
# Proporción de cantidad de registros en endurance 
df_usuario["share_endurance"] = ( 
    df_usuario["n_goal_endurance"] 
    / df_usuario["n_records"] 
) 
df_usuario["user_id"].nunique() 
df_usuario.shape[0]
df_usuario["user_id"].duplicated().sum() 
df_usuario["share_fat_loss"] = (
    df_usuario["n_goal_fat_loss"]
    / df_usuario["n_records"]
)

df_usuario["share_mobility"] = (
    df_usuario["n_goal_mobility"]
    / df_usuario["n_records"]
)

df_usuario["share_muscle_gain"] = (
    df_usuario["n_goal_muscle_gain"]
    / df_usuario["n_records"]
)
df_usuario = df_usuario.rename(columns={

    "user_id": "usuario",
    "age": "edad",

    "n_records": "num_entrenamientos",
    "n_exercises": "num_ejercicios_distintos",

    "n_bodyparts": "num_partes_cuerpo",
    "n_equipment": "num_tipos_equipo",
    "n_target_muscles": "num_musculos_trabajados",

    "mean_rating": "rating_promedio",
    "median_rating": "rating_mediana",
    "sd_rating": "variabilidad_rating",

    "n_goal_general_fitness": "veces_fitness_general",
    "n_goal_endurance": "veces_resistencia",
    "n_goal_fat_loss": "veces_perdida_grasa",
    "n_goal_mobility": "veces_movilidad",
    "n_goal_muscle_gain": "veces_ganancia_muscular",

    "share_general_fitness": "porc_fitness_general",
    "share_endurance": "porc_resistencia",
    "share_fat_loss": "porc_perdida_grasa",
    "share_mobility": "porc_movilidad",
    "share_muscle_gain": "porc_ganancia_muscular"
})


print("""Punto 1 y 2 correlacion""")
correlaciones = {
    "Partes cuerpo vs músculos trabajados":
        df_usuario["num_partes_cuerpo"].corr(
            df_usuario["num_musculos_trabajados"]
        ),

    "Rating promedio vs fitness general":
        df_usuario["rating_promedio"].corr(
            df_usuario["porc_fitness_general"]
        ),

    "Fitness general vs ganancia muscular":
        df_usuario["porc_fitness_general"].corr(
            df_usuario["porc_ganancia_muscular"]
        )
}

for nombre, valor in correlaciones.items():
    print(f"{nombre}: {valor:.3f}")
    
corr= df_usuario.select_dtypes(include=np.number).corr()
plt.figure(figsize=(12,8))
plt.imshow(corr, cmap="coolwarm")
plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.show()
