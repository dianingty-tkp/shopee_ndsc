def build_model(params, title_w2v_weights):
    # image model
    if params["image_model_type"]=="MobileNetV2":
        image_model = tf.keras.applications.MobileNetV2(input_shape=(params["image_size"], params["image_size"], 3),
                                                    include_top=False,
                                                    weights='imagenet'
                                                    )
    elif params["image_model_type"]=="InceptionResNetV2":
        image_model = tf.keras.applications.MobileNetV2(input_shape=(params["image_size"], params["image_size"], 3),
                                                    include_top=False,
                                                    weights='imagenet',
                                                    name=f"image_{idx}"
                                                    )

    # embedding
    product_embedding = tf.keras.layers.Embedding(
        title_w2v_weights.shape[0], output_dim=params["title_w2v_dim"], input_length=params["title_max_length"], weights=[title_w2v_weights],
        trainable=False)


    product_representation = []
    products_layer = {
        "product_1" : {},
        "product_2" : {},
    }

    for i in range(2):
        idx = i + 1

        if params["title_model_type"]=="CNN":
            products_layer[f"product_{idx}"]["input_title"] = tf.keras.Input(
                shape=(params["title_max_length"],), name=f"token_title_{idx}")

            # embedding
            products_layer[f"product_{idx}"]["t_embedding"] = product_embedding(products_layer[f"product_{idx}"]["input_title"])

            # kernel size 1
            products_layer[f"product_{idx}"]["t_x1"] = tf.keras.layers.Conv1D(
                filters=params["title_units"][0], kernel_size=params["title_kernel_size"][0], 
                activation='relu', name=f"conv1_{idx}")(products_layer[f"product_{idx}"]["t_embedding"])
            products_layer[f"product_{idx}"]["t_x1"] = tf.keras.layers.GlobalMaxPool1D(name=f"globalmax1_{idx}")(products_layer[f"product_{idx}"]["t_x1"])

            # kernel size 2
            products_layer[f"product_{idx}"]["t_x2"] = tf.keras.layers.Conv1D(
                filters=params["title_units"][1], kernel_size=params["title_kernel_size"][1], 
                activation='relu', name=f"conv2_{idx}")(products_layer[f"product_{idx}"]["t_embedding"])
            products_layer[f"product_{idx}"]["t_x2"] = tf.keras.layers.GlobalMaxPool1D(name=f"globalmax2_{idx}")(products_layer[f"product_{idx}"]["t_x2"])

            # kernel size 3
            products_layer[f"product_{idx}"]["t_x3"] = tf.keras.layers.Conv1D(
                filters=params["title_units"][2], kernel_size=params["title_kernel_size"][2], 
                activation='relu', name=f"conv3_{idx}")(products_layer[f"product_{idx}"]["t_embedding"])
            products_layer[f"product_{idx}"]["t_x3"] = tf.keras.layers.GlobalMaxPool1D(name=f"globalmax3_{idx}")(products_layer[f"product_{idx}"]["t_x3"])

            # concat
            products_layer[f"product_{idx}"]["t_concat"] = tf.keras.layers.concatenate(
                [products_layer[f"product_{idx}"]["t_x1"], products_layer[f"product_{idx}"]["t_x2"], products_layer[f"product_{idx}"]["t_x3"]])

            # output
            products_layer[f"product_{idx}"]["output_title"] = tf.keras.layers.Dropout(params["title_dropout_rate"])(products_layer[f"product_{idx}"]["t_concat"])
            products_layer[f"product_{idx}"]["output_title"] = tf.keras.layers.Dense(
                units=32, activation='relu')(products_layer[f"product_{idx}"]["output_title"])

        elif params["title_model_type"]=="BLSTM":
            products_layer[f"product_{idx}"]["input_title"] = tf.keras.Input(
                shape=(params["title_max_length"],), name=f"token_title_{i}")

            # embedding
            t_embedding = tf.keras.layers.Embedding(
                title_w2v_weights.shape[0], output_dim = params["title_w2v_dim"], input_length=params["title_max_length"], weights=[title_w2v_weights],
                trainable=False)(products_layer[f"product_{idx}"]["input_title"])

            # BLSTM
            products_layer[f"product_{idx}"]["t_blstm"] = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(params["title_units"][0], dropout=params["title_dropout_rate"], return_sequences=True))(t_embedding)
            products_layer[f"product_{idx}"]["t_blstm"] = tf.keras.layers.Reshape(
                target_shape=(params["title_max_length"], params["title_units"][0]*2,1))(products_layer[f"product_{idx}"]["t_blstm"])

            # 2D-CNN
            products_layer[f"product_{idx}"]["t_cnn"] = tf.keras.layers.Conv2D(filters=params["title_units"][1], kernel_size=params["title_kernel_size"][0])(products_layer[f"product_{idx}"]["t_blstm"])
            products_layer[f"product_{idx}"]["t_x"] = tf.keras.layers.MaxPooling2D(name=f'2DPooling_{idx}')(products_layer[f"product_{idx}"]["t_cnn"])
            products_layer[f"product_{idx}"]["t_x"] = tf.keras.layers.Flatten(name=f'Flatten_{idx}')(products_layer[f"product_{idx}"]["t_x"])

            # output
            products_layer[f"product_{idx}"]["t_x"] = tf.keras.layers.Dropout(params["title_dropout_rate"], name=f'Dropout_{idx}')(products_layer[f"product_{idx}"]["t_x"])
            products_layer[f"product_{idx}"]["output_title"] = tf.keras.layers.Dense(units=32, activation='relu')(products_layer[f"product_{idx}"]["t_x"])

        # base_model.trainable = False
        products_layer[f"product_{idx}"]["image_input"] = tf.keras.Input(
                shape=(160,160,3), name=f"byte_image_{idx}")
        products_layer[f"product_{idx}"]["i_x"] = image_model(products_layer[f"product_{idx}"]["image_input"])

        products_layer[f"product_{idx}"]["i_x"] = tf.keras.layers.GlobalMaxPooling2D()(products_layer[f"product_{idx}"]["i_x"])
        products_layer[f"product_{idx}"]["i_x"] = tf.keras.layers.Dropout(params["image_dropout_rate"])(products_layer[f"product_{idx}"]["i_x"])
        products_layer[f"product_{idx}"]["output_image"] = tf.keras.layers.Dense(32, activation='relu')(products_layer[f"product_{idx}"]["i_x"]) 

        products_layer[f"product_{idx}"]["prod_output"] = tf.keras.layers.concatenate(
            [products_layer[f"product_{idx}"]["output_title"], products_layer[f"product_{idx}"]["output_image"]])
        products_layer[f"product_{idx}"]["prod_output"] = tf.keras.layers.Dropout(
            params["global_dropout_rate"])(products_layer[f"product_{idx}"]["prod_output"])
        products_layer[f"product_{idx}"]["prod_output"] = tf.keras.layers.Dense(
            128, activation='relu')(products_layer[f"product_{idx}"]["prod_output"])

        product_representation.append(products_layer[f"product_{idx}"]["prod_output"])

    # concat all product representations
    concat_all = tf.keras.layers.concatenate(product_representation)
    output = tf.keras.layers.Dense(128, activation='relu')(concat_all)
    output = tf.keras.layers.Dropout(params["global_dropout_rate"])(output)
    output = tf.keras.layers.Dense(1, activation='sigmoid', name="Label")(output)
    
    inputs = [products_layer[f"product_1"]["input_title"], 
              products_layer[f"product_2"]["input_title"], 
              products_layer[f"product_1"]["image_input"], 
              products_layer[f"product_2"]["image_input"]]

    model = tf.keras.Model(inputs=inputs, outputs=output)

    base_learning_rate = params["learning_rate"]
    if params["optimizer"] == 'sgd':
        opt = tf.keras.optimizers.SGD(lr=base_learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
        print('Optimizer', params["optimizer"])
    else:
        opt = 'adam'
        print('Use default optimizer, Adam')

    model.compile(optimizer=opt,
                loss='binary_crossentropy',
                metrics=['accuracy', 'Recall', 'Precision']
                )
    return model