// src/main/java/com/davita/framework/logger/commands/CheckLoggerSession.java
package com.automationanywhere.botcommand.samples.commands.basic.GlobalSession;

import com.automationanywhere.botcommand.data.Value;
import com.automationanywhere.botcommand.data.impl.StringValue;
import com.automationanywhere.commandsdk.annotations.*;
import com.automationanywhere.commandsdk.annotations.rules.SessionAllowNonExistent;
import com.automationanywhere.commandsdk.annotations.rules.SessionObject;
import com.automationanywhere.commandsdk.model.AttributeType;
import com.automationanywhere.commandsdk.model.DataType;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.time.format.DateTimeFormatter;

import static com.automationanywhere.commandsdk.model.DataType.STRING;

@BotCommand
@CommandPkg(
        name = "CheckLoggerSession",
        label = "[[CheckLoggerSession.label]]",
        node_label = "[[CheckLoggerSession.nodeLabel]]",
        description = "[[CheckLoggerSession.desc]]",
        icon = "book.svg",
        text_color = "#0069b1",
        group_label = "[[Logger.group]]",
        comment = true,
        return_type = STRING,
        return_required = true
)

public class CheckLoggerSession1 {

    private static final Logger logger = LogManager.getLogger(CheckLoggerSession1.class);
    private static final DateTimeFormatter TS_FORMAT =
            DateTimeFormatter.ofPattern("yyyy-MM-dd'T'HH:mm:ss.SSSXXX");

    @GlobalSessionContext
    private com.automationanywhere.bot.service.GlobalSessionContext globalSessionContext;

    public void setGlobalSessionContext(com.automationanywhere.bot.service.GlobalSessionContext globalSessionContext) {
        this.globalSessionContext = globalSessionContext;
    }

    @Execute
    public Value<String> action(
            @Idx(index = "2", type = AttributeType.SESSION)
            @Pkg(label = "[[CheckLoggerSession.session.label]]", description = "[[CheckLoggerSession.session.desc]]",
                    default_value = "DefaultSession", default_value_type = DataType.SESSION)
            //Using the sessionObject annotation here as its a consumer class
            @SessionObject
//            @NotEmpty
            @SessionAllowNonExistent

            DemoForSession session
    )
//    {
//        return new StringValue(session.isClosed());
//    }
    {
        try {
            if (session == null || session.isClosed()) {
                return new StringValue("false");
            }


            logger.info("Logger session valid.");
            return new StringValue("true");

        } catch (Exception e) {
            logger.error("Error while validating logger session", e);
            return new StringValue("false from catch");
        }
    }
}