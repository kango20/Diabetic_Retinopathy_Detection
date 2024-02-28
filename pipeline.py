import preprocessing
import model
import evaluation
import visualization


datagen = preprocessing.Data_Gen()

model = model.Model(datagen.train_gen, datagen.valid_gen, datagen.test_gen)
eval = evaluation.Eval(datagen, model)
eval.data_size()
eval.evaluate_model()
viz = visualization.Viz(datagen, model)
viz.display_set_distributions()
viz.display_images_with_labels()
viz.plot_accuracy()
viz.plot_loss()
viz.gen_confusion_matrix()
viz.gen_classification_report()
viz.display_images_with_predictions()