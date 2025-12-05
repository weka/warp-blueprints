FROM nginx:alpine

# Copy static viewer files
COPY index.html /usr/share/nginx/html/index.html

# Nginx configuration to listen on 8080 and serve structure from /data
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 8080
CMD ["nginx", "-g", "daemon off;"]