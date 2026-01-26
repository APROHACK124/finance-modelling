# 01_preprocess_and_langid

## ¿Por qué limpiar texto y cuándo NO conviene?

Los modelos (BERT/Transformers, LLMs) son sensibles al **ruido** y al **formato**:
- **Corrección Unicode (ftfy):** Arregla caracteres rotos (�, mojibake) y normaliza comillas/acentos.  
  **Beneficio:** Evita que el modelo “vea” tokens extraños y mejora el match con vocabularios preentrenados.  
  **No conviene:** Casi siempre conviene. Evita si necesitas auditar *exactamente* el texto crudo.

- **Normalización de espacios/control:** Colapsa espacios repetidos y remueve caracteres invisibles.  
  **Beneficio:** Tokens consistentes, menos longitud.  
  **No conviene:** Cuando el *layout* o la cantidad de espacios es información (p. ej., ASCII tables).

- **URLs/usuarios/hashtags:** Reemplazar o estandarizar evita que cada URL/usuario sea un token raro.  
  **Beneficio:** Generaliza mejor (p. ej., `URL` o el dominio).  
  **No conviene:** Si el contenido exacto de la URL o el *handle* es una señal.

- **Emojis:** Pueden portar sentimiento. Convertirlos a `:smile:` (emoji.demojize) los hace visibles como texto.  
  **Beneficio:** Modelos “clásicos” y algunos tokenizadores capturan mejor la semántica.  
  **No conviene:** Si la visual del emoji es clave para UX (dashboards) o si el modelo ya maneja bien emojis.

- **Casing (mayúsculas/minúsculas):** Algunos modelos son *cased* (diferencian mayúsculas).  
  **Beneficio:** Mantener *case* conserva entidades y acrónimos.  
  **No conviene:** Si usas modelos *uncased* o *features* que no lo necesitan, puedes *lowercase* para robustez.

- **Puntuación/tokens repetidos/elongaciones:** Limitar ruido (ej. `cooooool`) puede ayudar.  
  **Beneficio:** Reduce OOV/fragmentación de tokens.  
  **No conviene:** Si las elongaciones contienen señal emocional útil para la tarea.

- **Detección de idioma:** Etiquetar idioma por documento/texto permite enrutar a modelos específicos y métricas por idioma.  

### Tres variantes de texto

- **text_ml:** para modelos clásicos/transformers. Minimiza ruido, preserva *casing* si trabajas con modelos *cased*.  
- **text_llm:** para *prompts*. Conserva emojis como `:smile:` y reemplaza URLs por el dominio (`DOMAIN_TOKEN`).  
- **text_viz:** para dashboards legibles. URLs acortadas al dominio visible, emojis en su forma visual.