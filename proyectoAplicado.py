# ==============================================================================
# PROYECTO JUSTIA - PRODUCTO MÍNIMO VIABLE (MVP)
# Módulo: Clasificación de Consultas Jurídicas, Extracción de Tokens y Explicabilidad
# Autor: Estudiante Posgrado - Corporación Universitaria de Asturias
# ==============================================================================

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline



# ------------------------------------------------------------------------------
# 1. CONSTRUCCIÓN DEL CORPUS DE ENTRENAMIENTO (SINTÉTICO - CONTEXTO COLOMBIANO)
# ------------------------------------------------------------------------------
corpus_justia = [
    # Derecho de Familia
    ("Solicito demanda de alimentos para mis dos hijos menores de edad, el padre no responde desde hace 6 meses.", "Familia"),
    ("Proceso de divorcio de mutuo acuerdo y liquidación de sociedad conyugal con bienes inmuebles.", "Familia"),
    ("Reglamentación de custodia y visitas para un niño de 5 años tras separación de hecho.", "Familia"),
    ("Violencia intrafamiliar y agresiones físicas por parte de mi cónyuge en el hogar.", "Familia"),
    ("Filiación extramatrimonial y prueba de ADN para reconocimiento de paternidad.", "Familia"),
    ("Fijación de cuota alimentaria e incremento por gastos escolares y médicos.", "Familia"),
    ("Embargo de salario por incumplimiento de la cuota alimentaria pactada en comisaría.", "Familia"),
    
    # Derecho Laboral
    ("Despido sin justa causa estando en estado de embarazo, solicito fuero de maternidad.", "Laboral"),
    ("No me han pagado las liquidaciones, cesantías ni prima de servicios de los últimos 2 años.", "Laboral"),
    ("Acoso laboral sistemático por parte del jefe inmediato con sobrecarga de tareas.", "Laboral"),
    ("Reclamación de pensión de invalidez por accidente laboral no cubierto por la ARL.", "Laboral"),
    ("Terminación unilateral del contrato de trabajo sin indemnización legal correspondiente.", "Laboral"),
    ("No pago de horas extras, recargos nocturnos y dominicales trabajados en la empresa.", "Laboral"),
    ("Cobro de fueros sindicales por despido injustificado durante negociación colectiva.", "Laboral"),

    # Derecho Penal
    ("Denuncia por hurto calificado y agravado de celular con arma blanca en la vía pública.", "Penal"),
    ("Víctima de violencia intrafamiliar y lesiones personales en concurso con amenaza.", "Penal"),
    ("Denuncia por estafa agravada y falsedad en documento privado en compraventa de vehículo.", "Penal"),
    ("Proceso penal por delitos sexuales e injuria por vías de hecho contra menor.", "Penal"),
    ("Investigación de la Fiscalía por estafa masiva mediante esquema piramidal.", "Penal"),
    ("Denuncia por injuria y calumnia en redes sociales con afectación al buen nombre.", "Penal"),

    # Derecho Civil
    ("Incumplimiento de contrato de arrendamiento de local comercial y restitución del inmueble.", "Civil"),
    ("Proceso ejecutivo por cobro de pagaré y letra de cambio vencida sin cancelar.", "Civil"),
    ("Demanda de pertenencia por prescripción adquisitiva del dominio de predio urbano.", "Civil"),
    ("Responsabilidad civil extracontractual por accidente de tránsito con daños materiales.", "Civil"),
    ("Cobro de honorarios profesionales por contrato de prestación de servicios incumplido.", "Civil"),
    ("Demanda por vicios ocultos en la venta de un apartamento nuevo con humedades.", "Civil"),

    # Derecho Constitucional / Tutela
    ("Acción de tutela por vulneración al derecho a la salud y no entrega de medicamentos vitales.", "Constitucional"),
    ("Tutela contra la EPS por negación de procedimiento quirúrgico ordenado por médico especialista.", "Constitucional"),
    ("Vulneración del derecho fundamental al debido proceso en procedimiento administrativo.", "Constitucional"),
    ("Acción de tutela para proteger derecho al mínimo vital de persona en condición de discapacidad.", "Constitucional"),
    ("Derecho de petición no respondido por la Alcaldía Municipal en el término legal de 15 días.", "Constitucional"),
    ("Tutela para protección de derecho a la educación de niño al que niegan matrícula.", "Constitucional")
]

# Crear DataFrame y balancear por réplica para la demo
df = pd.DataFrame(corpus_justia, columns=['texto', 'categoria'])
df_expanded = pd.concat([df] * 4, ignore_index=True)

# ------------------------------------------------------------------------------
# 2. ENTRENAMIENTO Y EVALUACIÓN DEL MODELO
# ------------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df_expanded['texto'], df_expanded['categoria'], test_size=0.25, random_state=42, stratify=df_expanded['categoria']
)

pipeline_justia = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
    ('clf', MultinomialNB(alpha=0.1))
])

pipeline_justia.fit(X_train, y_train)

# Métricas de rendimiento
y_pred = pipeline_justia.predict(X_test)
print("=================================================================")
print("             REPORTE DE EVALUACIÓN DEL MODELO JUSTIA              ")
print("=================================================================")
print(classification_report(y_test, y_pred))

# ------------------------------------------------------------------------------
# 3. MOTOR DE INFERENCIA, EXPLICABILIDAD Y TRAZABILIDAD (HUMAN-IN-THE-LOOP)
# ------------------------------------------------------------------------------
def inferencia_explicable_justia(texto_consulta, pipeline, umbral_certeza=0.60):
    """
    Recibe la consulta del usuario, evalúa el nivel de certeza del modelo,
    identifica tokens clave y genera la justificación jurisprudencial.
    """
    probs = pipeline.predict_proba([texto_consulta])[0]
    classes = pipeline.classes_
    best_idx = np.argmax(probs)
    best_class = classes[best_idx]
    confidence = probs[best_idx]
    
    # Extracción de n-gramas / tokens clave identificados por el TF-IDF
    tfidf_vec = pipeline.named_steps['tfidf']
    feature_names = np.array(tfidf_vec.get_feature_names_out())
    transformed = tfidf_vec.transform([texto_consulta]).toarray()[0]
    nonzero_indices = np.where(transformed > 0)[0]
    tokens_identificados = list(feature_names[nonzero_indices])
    
    # Mapeo de Fuentes Normativas Colombianas
    normativa_map = {
        "Familia": "Código de la Infancia y la Adolescencia (Ley 1098 de 2006) / Ley 1257 de 2008 / Código Civil.",
        "Laboral": "Código Sustantivo del Trabajo (CST) / Ley 50 de 1990 / Ley 1010 de 2006.",
        "Penal": "Código Penal Colombiano (Ley 599 de 2000) / Ley 906 de 2004.",
        "Civil": "Código Civil Colombiano / Código General del Proceso (Ley 1564 de 2012).",
        "Constitucional": "Constitución Política de Colombia (Art. 86 - Acción de Tutela) / Decreto 2591 de 1991."
    }
    
    # Evaluación del umbral de seguridad ética
    if confidence >= umbral_certeza:
        categoria_final = best_class
        estado_control = "APROBADO PARA REVISIÓN ESTUDIANTIL"
        alerta = "Sugerencia algorítmica confiable. Proceder a validación por el practicante."
    else:
        categoria_final = "REQUIERE REVISIÓN MANUAL DIRECTA (INCERTIDUMBRE)"
        estado_control = "BLOQUEO AUTOMÁTICO POR ALTA INCERTIDUMBRE"
        alerta = "Atención: La certeza es inferior al umbral ético (60%). Derivar a docente tutor."

    return {
        "Consulta_Usuario": texto_consulta,
        "Categoria_Sugerida": categoria_final,
        "Nivel_Certeza": f"{round(confidence * 100, 2)}%",
        "Tokens_Clave": tokens_identificados[:6],
        "Marco_Normativo_Asociado": normativa_map.get(best_class, "Normatividad General"),
        "Estado_Control_Etico": estado_control,
        "Mensaje_Alerta": alerta
    }

# ------------------------------------------------------------------------------
# 4. SIMULACIÓN DE PRUEBAS CON CASOS DE USO REALES Y FUERA DE DOMINIO
# ------------------------------------------------------------------------------
casos_prueba = [
    "Acción de tutela contra la EPS por no entregar medicamento oncológico vital.",
    "Despido sin justa causa estando en estado de embarazo sin permiso del inspector de trabajo.",
    "Hurto de celular mediante atraco con arma blanca en el centro de la ciudad.",
    "Quiero comprar una pizza con piña y pagar con tarjeta de crédito débito." # Fuera de dominio
]

print("\n=================================================================")
print("          DEMOSTRACIÓN DE FUNCIONAMIENTO - PANEL JUSTIA          ")
print("=================================================================")

for i, caso in enumerate(casos_prueba, 1):
    res = inferencia_explicable_justia(caso, pipeline_justia, umbral_certeza=0.60)
    print(f"\n--- CONSULTA {i} ---")
    print(f"Entrada: {res['Consulta_Usuario']}")
    print(f"Categoría Resultante: {res['Categoria_Sugerida']}")
    print(f"Certeza: {res['Nivel_Certeza']}")
    print(f"Tokens Clave Extraídos: {res['Tokens_Clave']}")
    print(f"Fuente Normativa: {res['Marco_Normativo_Asociado']}")
    print(f"Gobernanza Ética: {res['Estado_Control_Etico']} -> {res['Mensaje_Alerta']}")