from qgis.core import QgsProject, QgsVectorLayer, QgsVectorFileWriter
pathToFile = '<path-to-gml-dir>'
for vLayer in project.mapLayers().values():
    print(vLayer, pathToFile+vLayer.name())    
    QgsVectorFileWriter.writeAsVectorFormat(vLayer, pathToFile + vLayer.name() + ".geojson", "utf-8", vLayer.crs(), "GeoJson")
