import os
import re
import anthropic
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, text
from pydantic import BaseModel

app = FastAPI(
    title="ACME Natural Language API",
    description="Consultas en lenguaje natural sobre la base de datos ACME",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────
#  CONEXIÓN A MYSQL (igual que acme-api)
# ─────────────────────────────────────────
DATABASE_URL = os.environ["MYSQL_URL"]

if DATABASE_URL.startswith("mysql://"):
    DATABASE_URL = DATABASE_URL.replace("mysql://", "mysql+pymysql://", 1)

engine = create_engine(DATABASE_URL)

# ─────────────────────────────────────────
#  ESQUEMA DE LA BD PARA EL MODELO DE IA
# ─────────────────────────────────────────
DB_SCHEMA = """
Base de datos MySQL llamada 'railway' con las siguientes tablas:

1. clientes (Num_Clie, Empresa, Rep_Clie, Limite_Credito)
   - Num_Clie: identificador unico del cliente (PK)
   - Empresa: nombre de la empresa cliente
   - Rep_Clie: representante asignado (FK -> repventas.Num_Empl)
   - Limite_Credito: limite de credito en dolares

2. oficinas (Oficina, Ciudad, Region, Dir, Objetivo, Ventas)
   - Oficina: codigo de la oficina (PK)
   - Ciudad, Region: ubicacion
   - Dir: director (FK -> repventas.Num_Empl)
   - Objetivo: cuota objetivo, Ventas: ventas actuales

3. repventas (Num_Empl, Nombre, Edad, Oficina_Rep, Titulo, Contrato, Director, Cuota, Ventas)
   - Num_Empl: numero de empleado (PK)
   - Nombre: nombre completo
   - Oficina_Rep: oficina donde trabaja (FK -> oficinas.Oficina)
   - Cuota: cuota asignada, Ventas: ventas realizadas

4. pedidos (Num_Pedido, Fecha_Pedido, Clie, Rep, Fab, Producto, Cant, Importe)
   - Num_Pedido: numero de pedido (PK)
   - Clie: cliente (FK -> clientes.Num_Clie)
   - Rep: representante (FK -> repventas.Num_Empl)
   - Fab + Producto: producto (FK -> productos.Id_Fab + Id_Producto)
   - Cant: cantidad, Importe: monto total

5. productos (Id_Fab, Id_Producto, Descripcion, Precio, Costo, Existencias)
   - Id_Fab + Id_Producto: clave compuesta (PK)
   - Descripcion: nombre del producto
   - Precio: precio de venta, Costo: costo de compra
   - Existencias: stock disponible

Relaciones:
- clientes.Rep_Clie -> repventas.Num_Empl
- repventas.Oficina_Rep -> oficinas.Oficina
- pedidos.Clie -> clientes.Num_Clie
- pedidos.Rep -> repventas.Num_Empl
- pedidos.Fab + pedidos.Producto -> productos.Id_Fab + productos.Id_Producto
"""

# ─────────────────────────────────────────
#  MODELOS
# ─────────────────────────────────────────
class ConsultaRequest(BaseModel):
    pregunta: str

class SQLRequest(BaseModel):
    sql: str

# ─────────────────────────────────────────
#  FUNCIONES DE IA
# ─────────────────────────────────────────
def generar_sql(pregunta: str) -> str:
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    prompt = f"""Eres un experto en SQL para MySQL.
Dado este esquema:

{DB_SCHEMA}

Genera UNICAMENTE el SQL para responder: "{pregunta}"

Reglas:
- Solo SQL puro, sin explicaciones, sin markdown, sin ```
- Solo SELECT (nunca INSERT, UPDATE, DELETE, DROP)
- Usa LIMIT 50 si no se especifica cantidad
- Respeta los nombres exactos de columnas del esquema
"""
    msg = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}]
    )
    sql = msg.content[0].text.strip()
    return re.sub(r'```sql|```', '', sql).strip()


def generar_resumen(pregunta: str, resultado: list, sql: str) -> str:
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    muestra = resultado[:10]
    extra = f" (mostrando 10 de {len(resultado)})" if len(resultado) > 10 else ""
    prompt = f"""Eres analista de negocio de ACME.
Pregunta: "{pregunta}"
SQL: {sql}
Resultados{extra}: {muestra}

Escribe un resumen ejecutivo en espanol, maximo 3 oraciones."""
    msg = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}]
    )
    return msg.content[0].text.strip()

# ─────────────────────────────────────────
#  ENDPOINTS
# ─────────────────────────────────────────
@app.get("/")
def home():
    return {
        "status": "ACME NL API funcionando correctamente",
        "endpoints": {
            "GET  /health":   "Verifica conexion con la BD",
            "GET  /tablas":   "Lista tablas disponibles",
            "POST /consulta": "Consulta en lenguaje natural",
            "POST /sql":      "Ejecuta SQL directo (solo SELECT)",
            "GET  /docs":     "Documentacion Swagger UI"
        }
    }

@app.get("/health")
def health():
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"estado": "ok", "base_de_datos": "conectada"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tablas")
def tablas():
    return {
        "tablas": ["clientes", "oficinas", "repventas", "pedidos", "productos"],
        "descripcion": "Base de datos de ventas ACME"
    }

@app.post("/consulta")
def consulta_lenguaje_natural(request: ConsultaRequest):
    """
    Recibe una pregunta en lenguaje natural, genera SQL y devuelve
    resultados + resumen ejecutivo.

    Ejemplo body: {"pregunta": "Cuales son los 5 clientes con mas compras?"}
    """
    try:
        sql = generar_sql(request.pregunta)
        with engine.connect() as conn:
            rows = conn.execute(text(sql)).mappings().all()
        resultado = [dict(r) for r in rows]
        resumen = generar_resumen(request.pregunta, resultado, sql)
        return {
            "pregunta":     request.pregunta,
            "sql_generado": sql,
            "total":        len(resultado),
            "resultado":    resultado,
            "resumen":      resumen
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/sql")
def ejecutar_sql(request: SQLRequest):
    """Ejecuta SQL directo. Solo SELECT permitido."""
    if not request.sql.strip().lower().startswith("select"):
        raise HTTPException(status_code=400, detail="Solo se permiten consultas SELECT")
    try:
        with engine.connect() as conn:
            rows = conn.execute(text(request.sql)).mappings().all()
        resultado = [dict(r) for r in rows]
        return {"sql": request.sql, "total_registros": len(resultado), "resultado": resultado}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
