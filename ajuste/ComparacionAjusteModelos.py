
from Graficas import mostrarComparativaTodosModelos

# En este script se aplica una comparación entre todos modelos con los datos de estabilidad que se tienen guardados.
compararV6ConJuanjo = True # ¿Comparar entre nuestro mejor modelo y el modelo del artículo de referencia?

direccionTodosModelos = []
if not compararV6ConJuanjo:
    direccionTodosModelos.append({
        'nombre': 'V1',
        'direccionDatos': '../validacion/testEstabilidad_ModeloV1_22_08_2021_14_21.csv'
    })
    direccionTodosModelos.append({
        'nombre': 'V2',
        'direccionDatos': '../validacion/testEstabilidad_ModeloV2_22_08_2021_13_41.csv'
    })
    direccionTodosModelos.append({
        'nombre': 'V6',
        'direccionDatos': '../validacion/testEstabilidad_ModeloV6_17_08_2021_18_52.csv'
    })

    direccionTodosModelos.append({
        'nombre': 'V8',
        'direccionDatos': '../validacion/testEstabilidad_ModeloV8_22_08_2021_19_50.csv'
    })

else:
    direccionTodosModelos.append({
        # 'nombre': 'V6',
        'nombre': 'Proposed model',
        'direccionDatos': '../validacion/testEstabilidad_ModeloV6_17_08_2021_18_52.csv'
    })

mostrarComparativaTodosModelos(direccionTodosModelos, nombreWorkspace = '../workspc_30N', graficarModeloV0=True,
                               nombreFicheroExportado="ResultadosAjuste.csv")