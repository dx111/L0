import configparser
import numpy as np
import random

def generate_cfg(data,reg_type,sigma_value,lambda_scope):
    cfg = configparser.ConfigParser()
    section_name = "%s_%s_%s_%s"%(data,reg_type,str(sigma_value),str(lambda_scope[0]))
    cfg.add_section(section_name)
    # fixed-parameter
    if data=="mnist":
        cfg.set(section=section_name, option="dataset_name", value=data)
        cfg.set(section=section_name, option="input_shape", value="784")
        cfg.set(section=section_name, option="layer_node", value="400,300,100,10")
        cfg.set(section=section_name, option="train_epoch", value="300")
        cfg.set(section=section_name, option="base_learning_rate", value="1e-3")
        cfg.set(section=section_name, option="learning_decay", value="1e-3")
        cfg.set(section=section_name, option="batch_size", value="1024")
        cfg.set(section=section_name, option="dropout_rate", value="0.0")
        num_layer=len(np.array(cfg[section_name]["layer_node"].split(",")).astype("int"))
        # random choice hyperparameter
        cfg.set(section=section_name, option="reg_type", value=reg_type)
        reg_value=""
        for i in range(num_layer):
            if reg_value=="":
                reg_value+="%f"%lambda_scope[i]
            else:
                reg_value += ",%f"%lambda_scope[i]
        cfg.set(section=section_name, option="reg_lambda", value=reg_value)
        cfg.set(section=section_name, option="sigma", value="%f"%sigma_value)
    if data=="sdd":
        cfg.set(section=section_name, option="dataset_name", value=data)
        cfg.set(section=section_name, option="input_shape", value="48")
        cfg.set(section=section_name, option="layer_node", value="40,40,30,11")
        cfg.set(section=section_name, option="train_epoch", value="300")
        cfg.set(section=section_name, option="base_learning_rate", value="1e-3")
        cfg.set(section=section_name, option="learning_decay", value="1e-3")
        cfg.set(section=section_name, option="batch_size", value="1024")
        cfg.set(section=section_name, option="dropout_rate", value="0.0")
        num_layer=len(np.array(cfg[section_name]["layer_node"].split(",")).astype("int"))
        # random choice hyperparameter
        cfg.set(section=section_name, option="reg_type", value=reg_type)
        reg_value=""
        for i in range(num_layer):
            if reg_value=="":
                reg_value+="%f"%lambda_scope[i]
            else:
                reg_value += ",%f"%lambda_scope[i]
        cfg.set(section=section_name, option="reg_lambda", value=reg_value)
        cfg.set(section=section_name, option="sigma", value="%f"%sigma_value)
    if data=="covtype":
        cfg.set(section=section_name, option="dataset_name", value=data)
        cfg.set(section=section_name, option="input_shape", value="54")
        cfg.set(section=section_name, option="layer_node", value="50,50,20,7")
        cfg.set(section=section_name, option="train_epoch", value="300")
        cfg.set(section=section_name, option="base_learning_rate", value="1e-3")
        cfg.set(section=section_name, option="learning_decay", value="1e-3")
        cfg.set(section=section_name, option="batch_size", value="16384")
        cfg.set(section=section_name, option="dropout_rate", value="0.0")
        num_layer=len(np.array(cfg[section_name]["layer_node"].split(",")).astype("int"))
        # random choice hyperparameter
        cfg.set(section=section_name, option="reg_type", value=reg_type)
        reg_value=""
        for i in range(num_layer):
            if reg_value=="":
                reg_value+="%f"%lambda_scope[i]
            else:
                reg_value += ",%f"%lambda_scope[i]
        cfg.set(section=section_name, option="reg_lambda", value=reg_value)
        cfg.set(section=section_name, option="sigma", value="%f"%sigma_value)


    return cfg
