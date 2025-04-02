import json

# Nombre del archivo de entrada (JSONL) y salida (JSON array)
input_file = "dataset.jsonl"
output_file = "datos.json"

# Leer el archivo .jsonl y cargar cada línea como un objeto JSON
json_objects = []
with open(input_file, "r", encoding="utf-8") as file:
    for line in file:
        line = line.strip()  # Eliminar espacios y saltos de línea
        if line:  # Ignorar líneas vacías
            json_objects.append(json.loads(line))

# Guardar como un JSON array en un nuevo archivo
with open(output_file, "w", encoding="utf-8") as file:
    json.dump(json_objects, file, indent=2, ensure_ascii=False)

print(f"¡Conversión completada! Archivo guardado en: {output_file}")