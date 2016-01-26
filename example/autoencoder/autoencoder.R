require(mxnet)

AutoEncoder <-
  setRefClass(
    "AutoEncoder", fields = c(
      "data", "N", "dims", "stacks", "pt_dropout", "ft_dropout", "input_act",
      "internal_act", "output_act", "args", "args_grad", "args_mult", "auxs",
      "encoder", "internals", "decoder", "loss"
    )
  )

make_stack <-
  function(ctx, istack, data, num_input, num_hidden, sparseness_penalty = NULL,
           idropout = NULL, odropout = NULL,
           encoder_act = 'relu', decoder_act = 'relu') {
    x <- data
    if (!is.null(idropout)) {
      x <- mx.symbol.Dropout(data = x, p = idropout)
    }
    
    x = mx.symbol.FullyConnected(
      name = paste0('encoder_', istack), data = x, num.hidden = num_hidden
    )
    
    if (!is.null(encoder_act)) {
      x = mx.symbol.Activation(data = x, act.type = encoder_act)
      if (encoder_act == 'sigmoid' &&
          !is.null(sparseness_penalty)) {
        x <- mx.symbol.IdentityAttachKLSparseReg(
          data = x, name = paste0('sparse_encoder_', istack),
          penalty = sparseness_penalty
        )
        
      }
    }
    
    if (!is.null(odropout)) {
      x <- mx.symbol.Dropout(data = x, p = odropout)
    }
    
    x <- mx.symbol.FullyConnected(
      name = paste0('decoder_', istack),
      data = x, num.hidden = num_input
    )
    
    if (decoder_act == 'softmax') {
      x = mx.symbol.Softmax(
        data = x, label = data, prob.label = TRUE, act.type = decoder_act
      )
    } else if (!is.null(decoder_act)) {
      x = mx.symbol.Activation(data = x, act.type = decoder_act)
      if (decoder_act == 'sigmoid' &&
          !is.null(sparseness_penalty)) {
        x = mx.symbol.IdentityAttachKLSparseReg(
          data = x, name = paste0('sparse_decoder_', istack), penalty = sparseness_penalty
        )
      }
      x = mx.symbol.LinearRegressionOutput(data = x, label = data)
    } else {
      x = mx.symbol.LinearRegressionOutput(data = x, label = data)
    }
    
    args <- list()
    args[[paste0('encoder_', istack, '_weight')]] <-
      mxnet:::mx.nd.internal.empty(c(num_input, num_hidden), ctx)
    args[[paste0('encoder_', istack, '_bias')]] <-
      mxnet:::mx.nd.internal.empty(num_hidden, ctx)
    args[[paste0('decoder_', istack, '_weight')]] <-
      mxnet:::mx.nd.internal.empty(c(num_hidden, num_input), ctx)
    args[[paste0('decoder_', istack, '_bias')]] <-
      mxnet:::mx.nd.internal.empty(num_input, ctx)
    
    args_grad <- list()
    args_grad[[paste0('encoder_', istack, '_weight')]] <-
      mxnet:::mx.nd.internal.empty(c(num_input, num_hidden), ctx)
    args_grad[[paste0('encoder_', istack, '_bias')]] <-
      mxnet:::mx.nd.internal.empty(num_hidden, ctx)
    args_grad[[paste0('decoder_', istack, '_weight')]] <-
      mxnet:::mx.nd.internal.empty(c(num_hidden, num_input), ctx)
    args_grad[[paste0('decoder_', istack, '_bias')]] <-
      mxnet:::mx.nd.internal.empty(num_input, ctx)
    
    args_mult <- list()
    args_mult[[paste0('encoder_', istack, '_weight')]] <- 1.0
    args_mult[[paste0('encoder_', istack, '_bias')]] <- 2.0
    args_mult[[paste0('decoder_', istack, '_weight')]] <- 1.0
    args_mult[[paste0('decoder_', istack, '_bias')]] <- 2.0
    
    auxs <- list()
    
    if (encoder_act == 'sigmoid' && !is.null(sparseness_penalty)) {
      auxs[[paste0('sparse_encoder_', istack, '_moving_avg')]] = mx.nd.ones(num_hidden, ctx) * 0.5
    }
    
    if (decoder_act == 'sigmoid' && !is.null(sparseness_penalty)) {
      auxs[[paste0('sparse_decoder_', istack, '_moving_avg')]] = mx.nd.ones(num_input, ctx) * 0.5
    }
    
    init = mx.init.uniform(0.07)
    
    for (i in 1:length(names(args))) {
      init(names(args)[i], args[[i]], ctx)
    }
    
    return(
      list(
        'x' = x, 'args' = args, 'args_grad' = args_grad,
        'args_mult' = args_mult, 'auxs' = auxs
      )
    )
  }

make_encoder <-
  function(data, dims, sparseness_penalty = NULL, dropout = NULL,
           internal_act = 'relu', output_act = NULL) {
    x <- data
    internals <- list()
    N <- length(dims) - 1
    for (i in 1:N) {
      x <-
        mx.symbol.FullyConnected(
          name = paste0('encoder_', i), data = x, num_hidden = dims[i + 1]
        )
      if (!is.null(internal_act) && i < N) {
        x <- mmx.symbol.Activation(data = x, act.type = output_act)
        if (internal_act == "sigmod" &&
            !is.null(sparseness_penalty)) {
          x <-
            mx.symbol.IdentityAttachKLSparseReg(
              data = x, name = paste0('sparse_encoder_', i), penalty = sparseness_penalty
            )
        }
      } else if (!is.null(output_act) && i == N) {
        x <- mx.symbol.Activation(data = x, act.type = output_act)
        if (output_act == "sigmod" &&
            !is.null(sparseness_penalty)) {
          x <-
            mx.symbol.IdentityAttachKLSparseReg(
              data = x, name = paste0('sparse_encoder_', i), penalty = sparseness_penalty
            )
        }
      }
      
      if (!is.null(dropout)) {
        x <- mx.symbol.Dropout(data = x, p = dropout)
      }
      internals <- c(internals, x)
    }
    return(list('x' = x, 'internals' = internals))
  }


make_decoder <-
  function(feature, dims, sparseness_penalty = NULL, dropout = NULL, internal_act = 'relu', input_act = NULL) {
    x <- feature
    N = length(dims) - 1
    for (i in N:1) {
      x <-
        mx.symbol.FullyConnected(
          name = paste0('decoder_', i), data = x, num.hidden = dims[i]
        )
      if (!is.null(internal_act) && i > 1) {
        x <- mx.symbol.Activation(data = x, act.type = internal_act)
        if (internal_act == "sigmod" &&
            !is.null(sparseness_penalty)) {
          x <-
            mx.symbol.IdentityAttachKLSparseReg(
              data = x, name = paste0('sparse_decoder_', i), penalty = sparseness_penalty
            )
        }
      } else if (!is.null(input_act) && i == 1) {
        x <- mx.symbol.Activation(data = x, act.type = input_act)
        if (input_act == "sigmod" && !is.null(sparseness_penalty)) {
          x <-
            mx.symbol.IdentityAttachKLSparseReg(
              data = x, name = paste0('sparse_decoder_', i), penalty = sparseness_penalty
            )
        }
      }
      
      if (!is.null(dropout) && i > 1) {
        x <- mx.symbol.Dropout(data = x, p = dropout)
      }
    }
    return(x)
  }

AE_setup <-
  function(ctx, dims, sparseness_penalty = NULL, pt_dropout = NULL, ft_dropout = NULL,
           input_act = NULL, internal_act = 'relu', output_act = NULL) {
    data = mx.symbol.Variable('data')
    ae_model <- AutoEncoder$new('data' = data)
    ae_model$N <- length(dims) - 1
    ae_model$dims <- dims
    ae_model$stacks <- list()
    ae_model$pt_dropout <- pt_dropout
    ae_model$ft_dropout <- ft_dropout
    ae_model$input_act <- input_act
    ae_model$internal_act <- internal_act
    ae_model$output_act <- output_act
    
    for (i in 1:ae_model$N) {
      if (i == 1) {
        decoder_act = input_act
        idropout = NULL
      } else {
        decoder_act = internal_act
        idropout = pt_dropout
      }
      
      if (i == ae_model$N) {
        encoder_act = output_act
        odropout = NULL
      } else {
        encoder_act = internal_act
        odropout = pt_dropout
      }
      
      istack <-
        make_stack(
          ctx, i, ae_model$data, dims[i], dims[i + 1],
          sparseness_penalty, idropout, odropout, encoder_act, decoder_act
        )
      ae_model$stacks <- c(ae_model$stacks, istack$x)
      ae_model$args <- c(ae_model$args, istack$args)
      ae_model$args_grad <- c(ae_model$args_grad, istack$args_grad)
      ae_model$args_mult <- c(ae_model$args_mult, istack$args_mult)
      ae_model$auxs <- c(ae_model$auxs, istack$auxs)
    }
    encoder <-
      make_encoder(ae_model$data, dims, sparseness_penalty, ft_dropout, internal_act, output_act)
    ae_model$encoder <- encoder$x
    ae_model$internals <- encoder$internals
    
    decoder <-
      make_decoder(ae_model$encoder, dims, sparseness_penalty, ft_dropout, internal_act, input_act)
    if (input_act == "softmax") {
      ae_model$loss = ae_model$decoder
    } else {
      ae_model$loss <-
        mx.symbol.LinearRegressionOutput(data = ae_model$decoder, label = ae_model$data)
    }
  }

layerwise_pretrain <-
  function(ae_model, X, batch_size, n_iter, optimizer, l_rate, decay, lr_scheduler = NULL) {
    
  }