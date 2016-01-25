require(mxnet)

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